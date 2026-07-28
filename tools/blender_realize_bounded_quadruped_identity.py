#!/usr/bin/env python3
"""Audit or realize a reviewed bounded identity edit on one animated quadruped.

Audit mode imports an SHA-locked GLB, verifies the exact scene inventory and
explicit mirrored ear masks, predicts the bounded edit, and publishes an
authenticated *non-executable* preflight receipt.  It never mutates the asset.

Realize mode repeats the audit and additionally requires a separately hashed,
task-scoped machine-execution authorization bound to that preflight receipt.
Only then may it:

* rotate the two explicit ear vertex masks with mirrored transforms;
* remove one exact unskinned, material-free object;
* generate deterministic Smart-UV coordinates and a profile-driven two-colour
  embedded coat texture; and
* export/read back a research-candidate GLB while proving topology, weights,
  skeleton, and actions were unchanged in Blender memory.

Neither mode can authorize dataset registration, UE import, or Native merge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Mapping, Sequence

import bpy
from mathutils import Matrix


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import bounded_quadruped_identity_contract as contract  # noqa: E402


GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
POSITION_PRECISION = 6
INVARIANT_PRECISION = 8


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "realize"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--execution-authorization-sha256")
    args = parser.parse_args(argv)
    realize_only = (
        args.preflight,
        args.preflight_sha256,
        args.execution_authorization,
        args.execution_authorization_sha256,
    )
    if args.mode == "audit" and any(value is not None for value in realize_only):
        parser.error("audit mode forbids realization-only evidence")
    if args.mode == "realize" and any(value is None for value in realize_only):
        parser.error(
            "realize mode requires preflight and machine-execution "
            "authorization evidence"
        )
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def evidence_file_record(path: Path) -> dict[str, Any]:
    record = relative_file_record(path)
    record["path"] = f"evidence/{path.name}"
    return record


def authenticated_bytes(
    raw_path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    expected = contract.require_sha256(expected_sha256, f"{label} SHA-256")
    path = raw_path.absolute()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"missing or unsafe {label}: {path}") from error
    try:
        metadata_before = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_before.st_mode) or metadata_before.st_size <= 0:
            raise RuntimeError(f"{label} must be a non-empty regular file")
        chunks = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        metadata_after = os.fstat(descriptor)
        if (
            metadata_before.st_dev != metadata_after.st_dev
            or metadata_before.st_ino != metadata_after.st_ino
            or metadata_before.st_size != metadata_after.st_size
            or metadata_before.st_mtime_ns != metadata_after.st_mtime_ns
        ):
            raise RuntimeError(f"{label} identity changed while it was read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != metadata_before.st_size:
        raise RuntimeError(f"{label} size changed while it was read")
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return path, payload


def rounded(value: float, precision: int = INVARIANT_PRECISION) -> float:
    return round(float(value), precision)


def matrix_record(matrix) -> list[float]:
    return [rounded(value) for row in matrix for value in row]


def hash_record(value: Any) -> str:
    return contract.canonical_sha256(value)


def mesh_topology_sha256(mesh) -> str:
    return hash_record(
        {
            "vertices": len(mesh.vertices),
            "edges": sorted(
                sorted(int(index) for index in edge.vertices) for edge in mesh.edges
            ),
            "polygons": [
                {
                    "vertices": [int(index) for index in polygon.vertices],
                    "material_index": int(polygon.material_index),
                }
                for polygon in mesh.polygons
            ],
        }
    )


def vertex_position_records(mesh) -> list[dict[str, Any]]:
    return [
        {
            "index": int(vertex.index),
            "position": [rounded(value) for value in vertex.co],
        }
        for vertex in mesh.vertices
    ]


def vertex_position_sha256(mesh, indices=None) -> str:
    selected = None if indices is None else set(indices)
    return hash_record(
        [
            record
            for record in vertex_position_records(mesh)
            if selected is None or record["index"] in selected
        ]
    )


def vertex_weight_records(mesh_object) -> list[dict[str, Any]]:
    groups = mesh_object.vertex_groups
    return [
        {
            "index": int(vertex.index),
            "weights": sorted(
                [
                    {
                        "group": groups[item.group].name,
                        "weight": rounded(item.weight),
                    }
                    for item in vertex.groups
                ],
                key=lambda value: value["group"],
            ),
        }
        for vertex in mesh_object.data.vertices
    ]


def vertex_weight_sha256(mesh_object) -> str:
    return hash_record(vertex_weight_records(mesh_object))


def skeleton_sha256(
    armature_object,
    *,
    precision: int = INVARIANT_PRECISION,
) -> str:
    return hash_record(
        [
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "use_deform": bool(bone.use_deform),
                "matrix_local": [
                    rounded(value, precision)
                    for row in bone.matrix_local
                    for value in row
                ],
            }
            for bone in sorted(armature_object.data.bones, key=lambda item: item.name)
        ]
    )


def skeleton_hierarchy_sha256(armature_object) -> str:
    return hash_record(
        [
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "use_deform": bool(bone.use_deform),
            }
            for bone in sorted(armature_object.data.bones, key=lambda item: item.name)
        ]
    )


def action_sha256(*, precision: int = INVARIANT_PRECISION) -> str:
    records = []
    for action in sorted(bpy.data.actions, key=lambda item: item.name):
        curves = []
        for curve in sorted(
            action.fcurves,
            key=lambda item: (item.data_path, int(item.array_index)),
        ):
            curves.append(
                {
                    "data_path": curve.data_path,
                    "array_index": int(curve.array_index),
                    "keyframes": [
                        {
                            "co": [rounded(value, precision) for value in point.co],
                            "interpolation": point.interpolation,
                            "handle_left_type": point.handle_left_type,
                            "handle_right_type": point.handle_right_type,
                        }
                        for point in curve.keyframe_points
                    ],
                }
            )
        records.append(
            {
                "name": action.name,
                "frame_range": [
                    rounded(value, precision) for value in action.frame_range
                ],
                "curves": curves,
            }
        )
    return hash_record(records)


def mesh_inventory() -> list[dict[str, Any]]:
    records = []
    for item in sorted(bpy.data.objects, key=lambda value: (value.type, value.name)):
        record: dict[str, Any] = {
            "name": item.name,
            "type": item.type,
            "parent": item.parent.name if item.parent is not None else None,
        }
        if item.type == "MESH":
            modifiers = [
                modifier.object.name
                for modifier in item.modifiers
                if modifier.type == "ARMATURE" and modifier.object is not None
            ]
            record.update(
                {
                    "vertices": len(item.data.vertices),
                    "edges": len(item.data.edges),
                    "faces": len(item.data.polygons),
                    "materials": [
                        material.name if material is not None else None
                        for material in item.data.materials
                    ],
                    "armature_modifiers": sorted(modifiers),
                    "uv_layers": [layer.name for layer in item.data.uv_layers],
                }
            )
        elif item.type == "ARMATURE":
            record["bones"] = len(item.data.bones)
        records.append(record)
    return records


def object_geometry_sha256(mesh_object) -> str:
    return hash_record(
        {
            "name": mesh_object.name,
            "vertices": [
                [rounded(value) for value in vertex.co]
                for vertex in mesh_object.data.vertices
            ],
            "polygons": [
                [int(index) for index in polygon.vertices]
                for polygon in mesh_object.data.polygons
            ],
            "matrix_world": matrix_record(mesh_object.matrix_world),
        }
    )


def expected_actions(plan: contract.IdentityPlan) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for expected in plan.expected_actions:
        candidates = [
            action.name
            for action in bpy.data.actions
            if expected.lower() in action.name.lower()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one action matching {expected!r}, got {sorted(candidates)}"
            )
        resolved[expected] = candidates[0]
    if len(bpy.data.actions) != len(plan.expected_actions):
        raise RuntimeError("unexpected additional Blender actions are present")
    return resolved


def resolve_scene(plan: contract.IdentityPlan):
    primary = bpy.data.objects.get(plan.primary_mesh)
    armature = bpy.data.objects.get(plan.armature)
    removal = bpy.data.objects.get(plan.removal_object)
    if primary is None or primary.type != "MESH":
        raise RuntimeError(f"primary mesh is missing: {plan.primary_mesh}")
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"armature is missing: {plan.armature}")
    if removal is None or removal.type != "MESH":
        raise RuntimeError(f"removal object is missing: {plan.removal_object}")
    modifiers = [
        modifier
        for modifier in primary.modifiers
        if modifier.type == "ARMATURE" and modifier.object is armature
    ]
    if len(modifiers) != 1:
        raise RuntimeError("primary mesh must have one exact armature modifier")
    if (
        removal.parent is not None
        or any(modifier.type == "ARMATURE" for modifier in removal.modifiers)
        or len(removal.data.vertices) != plan.removal_vertices
        or len(removal.data.polygons) != plan.removal_faces
        or len(removal.data.materials) != 0
    ):
        raise RuntimeError(
            "removal target is not the exact unskinned material-free object"
        )
    custom_shape_bones = [
        bone for bone in armature.pose.bones if bone.custom_shape is not None
    ]
    if len(custom_shape_bones) != plan.removal_custom_shape_assignments or any(
        bone.custom_shape is not removal for bone in custom_shape_bones
    ):
        raise RuntimeError(
            "removal target is not the only expected pose-bone custom shape"
        )
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    if armatures != [armature] or set(meshes) != {primary, removal}:
        raise RuntimeError("source scene contains an unexpected armature or mesh")
    if len(primary.data.uv_layers) != 0 or len(bpy.data.images) != 0:
        raise RuntimeError(
            "this derivation route requires a source with no UV or image authority"
        )
    expected_actions(plan)
    return primary, armature, removal


def position_key(point: Sequence[float]) -> tuple[float, float, float]:
    return tuple(round(float(value), POSITION_PRECISION) for value in point)


def mask_groups(
    mesh, indices: Sequence[int]
) -> dict[tuple[float, float, float], list[int]]:
    groups: dict[tuple[float, float, float], list[int]] = {}
    for index in indices:
        if index >= len(mesh.vertices):
            raise RuntimeError(f"ear mask vertex index is out of range: {index}")
        key = position_key(mesh.vertices[index].co)
        groups.setdefault(key, []).append(index)
    return {key: sorted(value) for key, value in groups.items()}


def all_position_groups(mesh) -> dict[tuple[float, float, float], list[int]]:
    groups: dict[tuple[float, float, float], list[int]] = {}
    for vertex in mesh.vertices:
        groups.setdefault(position_key(vertex.co), []).append(vertex.index)
    return {key: sorted(value) for key, value in groups.items()}


def pair_mirrored_groups(
    positive: Mapping[tuple[float, float, float], list[int]],
    negative: Mapping[tuple[float, float, float], list[int]],
    *,
    plane_y: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    unused = set(negative)
    pairs = []
    for positive_position in sorted(positive):
        target = (
            positive_position[0],
            2.0 * plane_y - positive_position[1],
            positive_position[2],
        )
        candidates = [
            item
            for item in unused
            if max(abs(item[index] - target[index]) for index in range(3)) <= tolerance
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "explicit ear masks do not have one-to-one mirrored positions"
            )
        negative_position = candidates[0]
        unused.remove(negative_position)
        if len(positive[positive_position]) != len(negative[negative_position]):
            raise RuntimeError(
                "mirrored ear positions have different seam multiplicity"
            )
        pairs.append(
            {
                "positive_position": list(positive_position),
                "positive_indices": positive[positive_position],
                "negative_position": list(negative_position),
                "negative_indices": negative[negative_position],
            }
        )
    if unused:
        raise RuntimeError("negative ear mask contains unmatched mirrored positions")
    return pairs


def bbox(mesh) -> tuple[list[float], list[float], float]:
    minimum = [
        min(float(vertex.co[index]) for vertex in mesh.vertices) for index in range(3)
    ]
    maximum = [
        max(float(vertex.co[index]) for vertex in mesh.vertices) for index in range(3)
    ]
    diagonal = math.sqrt(
        sum((maximum[index] - minimum[index]) ** 2 for index in range(3))
    )
    if diagonal <= 0.0 or not math.isfinite(diagonal):
        raise RuntimeError("primary mesh has degenerate bounds")
    return minimum, maximum, diagonal


def audit_core(
    plan: contract.IdentityPlan,
    primary,
    armature,
    removal,
) -> dict[str, Any]:
    mesh = primary.data
    selected = set(plan.positive.indices) | set(plan.negative.indices)
    if len(selected) / len(mesh.vertices) > plan.maximum_moved_vertex_fraction:
        raise RuntimeError("explicit ear masks exceed the moved-vertex fraction gate")
    all_groups = all_position_groups(mesh)
    positive_groups = mask_groups(mesh, plan.positive.indices)
    negative_groups = mask_groups(mesh, plan.negative.indices)
    for key, group in {**positive_groups, **negative_groups}.items():
        if group != all_groups[key]:
            raise RuntimeError(
                "ear masks omit duplicate vertices at one selected position"
            )
    if any(position[1] <= plan.symmetry_plane_y for position in positive_groups) or any(
        position[1] >= plan.symmetry_plane_y for position in negative_groups
    ):
        raise RuntimeError("ear masks cross the declared symmetry plane")
    pairs = pair_mirrored_groups(
        positive_groups,
        negative_groups,
        plane_y=plan.symmetry_plane_y,
        tolerance=plan.symmetry_tolerance,
    )
    minimum, maximum, diagonal = bbox(mesh)
    predicted = {}
    maximum_displacement = 0.0
    for side in (plan.positive, plan.negative):
        for index in side.indices:
            source = tuple(float(value) for value in mesh.vertices[index].co)
            output = contract.rotate_ear_point(source, side)
            displacement = math.dist(source, output)
            maximum_displacement = max(maximum_displacement, displacement)
            predicted[index] = output
    if maximum_displacement / diagonal > plan.maximum_displacement_diagonal_ratio:
        raise RuntimeError("predicted ear displacement exceeds the diagonal gate")
    for pair in pairs:
        positive_output = predicted[pair["positive_indices"][0]]
        negative_output = predicted[pair["negative_indices"][0]]
        mirrored = (
            positive_output[0],
            2.0 * plan.symmetry_plane_y - positive_output[1],
            positive_output[2],
        )
        if (
            max(abs(mirrored[index] - negative_output[index]) for index in range(3))
            > plan.symmetry_tolerance * 2.0
        ):
            raise RuntimeError("predicted ear outputs are not symmetric")
    return {
        "source_inventory": mesh_inventory(),
        "primary_mesh": {
            "name": primary.name,
            "topology_sha256": mesh_topology_sha256(mesh),
            "vertex_position_sha256": vertex_position_sha256(mesh),
            "vertex_weight_sha256": vertex_weight_sha256(primary),
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
            "materials": [
                material.name if material is not None else None
                for material in mesh.materials
            ],
            "uv_layer_count": len(mesh.uv_layers),
            "bbox_minimum": minimum,
            "bbox_maximum": maximum,
            "bbox_diagonal": diagonal,
        },
        "armature": {
            "name": armature.name,
            "bone_count": len(armature.data.bones),
            "skeleton_sha256": skeleton_sha256(armature),
        },
        "actions": {
            "resolved": expected_actions(plan),
            "action_sha256": action_sha256(),
        },
        "removal_target": {
            "name": removal.name,
            "geometry_sha256": object_geometry_sha256(removal),
            "vertices": len(removal.data.vertices),
            "faces": len(removal.data.polygons),
            "unskinned": True,
            "material_free": True,
            "pose_bone_custom_shape_assignments": [
                bone.name
                for bone in sorted(
                    armature.pose.bones,
                    key=lambda value: value.name,
                )
                if bone.custom_shape is removal
            ],
            "only_pose_bone_custom_shape": True,
        },
        "ear_masks": {
            "positive_y": {
                "vertex_count": len(plan.positive.indices),
                "unique_position_count": len(positive_groups),
                "position_sha256": vertex_position_sha256(mesh, plan.positive.indices),
                "weight_sha256": hash_record(
                    [
                        vertex_weight_records(primary)[index]
                        for index in plan.positive.indices
                    ]
                ),
            },
            "negative_y": {
                "vertex_count": len(plan.negative.indices),
                "unique_position_count": len(negative_groups),
                "position_sha256": vertex_position_sha256(mesh, plan.negative.indices),
                "weight_sha256": hash_record(
                    [
                        vertex_weight_records(primary)[index]
                        for index in plan.negative.indices
                    ]
                ),
            },
            "mirrored_pair_count": len(pairs),
            "mirrored_pairs_sha256": hash_record(pairs),
            "all_duplicate_position_vertices_included": True,
            "disjoint": True,
        },
        "predicted_edit": {
            "moved_vertex_count": len(selected),
            "moved_vertex_fraction": len(selected) / len(mesh.vertices),
            "maximum_displacement": maximum_displacement,
            "maximum_displacement_bbox_diagonal_ratio": (
                maximum_displacement / diagonal
            ),
            "predicted_position_sha256": hash_record(
                [
                    {
                        "index": index,
                        "position": [rounded(value) for value in predicted[index]],
                    }
                    for index in sorted(predicted)
                ]
            ),
            "mirrored_output_positions": True,
        },
    }


def import_snapshot(snapshot: Path, *, disable_bone_shape: bool = False) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(
        filepath=str(snapshot),
        disable_bone_shape=disable_bone_shape,
    )
    if "FINISHED" not in result:
        raise RuntimeError("Blender GLB import did not finish")


def toolchain_record() -> dict[str, Any]:
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii", errors="replace")
    return {
        "blender": {
            "version": bpy.app.version_string,
            "build_hash": str(build_hash),
            "binary": str(Path(bpy.app.binary_path).absolute()),
        },
        "executor": file_record(Path(__file__).absolute()),
        "contract": file_record(Path(contract.__file__).absolute()),
    }


def snapshot_toolchain(
    publication: contract.SecurePublication,
) -> dict[str, Any]:
    snapshots = {
        "executor": publication.write_evidence(
            "blender_realize_bounded_quadruped_identity.py",
            Path(__file__).read_bytes(),
        ),
        "contract": publication.write_evidence(
            "bounded_quadruped_identity_contract.py",
            Path(contract.__file__).read_bytes(),
        ),
    }
    records = {name: evidence_file_record(path) for name, path in snapshots.items()}
    current = toolchain_record()
    if any(records[name]["sha256"] != current[name]["sha256"] for name in snapshots):
        raise RuntimeError("toolchain changed while its evidence was snapshotted")
    return records


def require_snapshot_publication_records(
    publication_records: Mapping[str, Mapping[str, Any]],
    expected_records: Mapping[str, Mapping[str, Any]],
) -> None:
    for relative_path, expected in expected_records.items():
        if publication_records.get(relative_path) != expected:
            raise RuntimeError(
                f"sealed snapshot no longer matches authenticated bytes: "
                f"{relative_path}"
            )


def audit_receipt(
    *,
    source_path: Path,
    source_payload: bytes,
    plan_path: Path,
    plan_payload: bytes,
    plan: contract.IdentityPlan,
    core: Mapping[str, Any],
    toolchain_snapshots: Mapping[str, Any],
    source_snapshot_path: Path,
    plan_snapshot_path: Path,
) -> dict[str, Any]:
    return {
        "schema": contract.PREFLIGHT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_machine_execution_authorization",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "execution_authorized": False,
        "source": {
            "path": str(source_path),
            "sha256": sha256_bytes(source_payload),
            "size_bytes": len(source_payload),
            "publication_snapshot": evidence_file_record(source_snapshot_path),
        },
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_bytes(plan_payload),
            "size_bytes": len(plan_payload),
            "canonical_sha256": contract.canonical_sha256(plan.raw),
            "publication_snapshot": evidence_file_record(plan_snapshot_path),
        },
        "toolchain": toolchain_record(),
        "toolchain_publication_snapshots": toolchain_snapshots,
        "publication_security_boundary": contract.publication_security_boundary(),
        "audit_core": core,
        "audit_core_sha256": contract.canonical_sha256(core),
        "automatic_checks": {
            "authenticated_source_and_plan": True,
            "exact_scene_inventory": True,
            "removal_target_is_unskinned_and_material_free": True,
            "explicit_masks_are_disjoint": True,
            "all_duplicate_position_vertices_are_included": True,
            "source_masks_are_mirrored": True,
            "predicted_outputs_are_mirrored": True,
            "moved_vertex_fraction_is_bounded": True,
            "displacement_is_bounded": True,
            "source_has_no_uv_or_image_authority": True,
            "overall": "passed_preflight_only",
        },
        "authority_boundary": {
            "preflight_is_not_a_user_decision": True,
            "machine_execution_authorization_required": True,
            "authorization_creates_research_candidate_only": True,
            "user_approval_inferred": False,
            "ue_import_authorized": False,
            "native_merge_authorized": False,
        },
        "next_gate": "task_scoped_machine_execution_authorization",
    }


def validate_coat_png_codes(
    payload: bytes,
    plan: contract.IdentityPlan,
    *,
    label: str,
) -> dict[str, Any]:
    codes = contract.png_rgba8_code_values(payload)
    expected_base = contract.srgb_rgba8(plan.base_color)
    expected_white = contract.srgb_rgba8(plan.white_color)
    if codes != {expected_base, expected_white}:
        raise RuntimeError(
            f"{label} PNG sRGB code values changed: "
            f"expected={sorted((expected_base, expected_white))} "
            f"actual={sorted(codes)}"
        )
    return {
        "encoding": "PNG_RGBA8_sRGB_code_values",
        "base_rgba8": list(expected_base),
        "white_rgba8": list(expected_white),
        "unique_rgba8_values": [list(value) for value in sorted(codes)],
        "sRGB_transfer_applied_exactly_once": True,
    }


def generate_uv(primary, plan: contract.IdentityPlan) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    primary.select_set(True)
    bpy.context.view_layer.objects.active = primary
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    result = bpy.ops.uv.smart_project(
        angle_limit=plan.uv_angle_limit,
        island_margin=plan.uv_island_margin,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    if "FINISHED" not in result or primary.data.uv_layers.active is None:
        raise RuntimeError("deterministic Smart UV projection did not finish")


def polygon_role_counts(
    primary,
    plan: contract.IdentityPlan,
) -> tuple[list[str], dict[str, int]]:
    mesh = primary.data
    minimum, maximum, _diagonal = bbox(mesh)
    roles = []
    for polygon in mesh.polygons:
        center = sum(
            (mesh.vertices[index].co for index in polygon.vertices),
            start=mesh.vertices[polygon.vertices[0]].co * 0.0,
        )
        center /= len(polygon.vertices)
        roles.append(
            contract.coat_role(
                center,
                minimum,
                maximum,
                plan.symmetry_plane_y,
                plan.coat_regions,
            )
        )
    counts = {role: roles.count(role) for role in ("base", "white")}
    if min(counts.values()) <= 0:
        raise RuntimeError("deterministic coat classified no faces for one colour")
    return roles, counts


def rasterize_coat_texture(
    primary,
    plan: contract.IdentityPlan,
    texture_path: Path,
) -> dict[str, Any]:
    import numpy as np

    mesh = primary.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("coat rasterization requires an active UV layer")
    roles, counts = polygon_role_counts(primary, plan)
    size = plan.uv_resolution
    # Blender's direct PNG save treats Image.pixels as the file's encoded channel
    # values.  The plan fields are already sRGB, so pre-linearizing here would
    # apply the transfer curve twice after a texture consumer performs its normal
    # sRGB decode.
    base = np.asarray(plan.base_color, dtype=np.float32)
    white = np.asarray(plan.white_color, dtype=np.float32)
    image_data = np.empty((size, size, 4), dtype=np.float32)
    image_data[:] = base
    white_mask = np.zeros((size, size), dtype=np.bool_)

    def edge(left, right, point):
        return (point[0] - left[0]) * (right[1] - left[1]) - (point[1] - left[1]) * (
            right[0] - left[0]
        )

    for polygon, role in zip(mesh.polygons, roles):
        if role != "white":
            continue
        loops = list(polygon.loop_indices)
        coordinates = [
            (
                float(uv_layer.data[index].uv.x) * (size - 1),
                (1.0 - float(uv_layer.data[index].uv.y)) * (size - 1),
            )
            for index in loops
        ]
        for offset in range(1, len(coordinates) - 1):
            triangle = (coordinates[0], coordinates[offset], coordinates[offset + 1])
            x_min = max(0, int(math.floor(min(point[0] for point in triangle))))
            x_max = min(
                size - 1,
                int(math.ceil(max(point[0] for point in triangle))),
            )
            y_min = max(0, int(math.floor(min(point[1] for point in triangle))))
            y_max = min(
                size - 1,
                int(math.ceil(max(point[1] for point in triangle))),
            )
            area = edge(triangle[0], triangle[1], triangle[2])
            if abs(area) <= 1.0e-12:
                raise RuntimeError("Smart UV produced a degenerate coat triangle")
            for y_value in range(y_min, y_max + 1):
                for x_value in range(x_min, x_max + 1):
                    sample = (x_value + 0.5, y_value + 0.5)
                    values = (
                        edge(triangle[1], triangle[2], sample),
                        edge(triangle[2], triangle[0], sample),
                        edge(triangle[0], triangle[1], sample),
                    )
                    if all(value >= 0.0 for value in values) or all(
                        value <= 0.0 for value in values
                    ):
                        white_mask[y_value, x_value] = True
    if not np.any(white_mask):
        raise RuntimeError("coat texture contains no rasterized white texels")
    for _iteration in range(3):
        expanded = white_mask.copy()
        expanded[1:, :] |= white_mask[:-1, :]
        expanded[:-1, :] |= white_mask[1:, :]
        expanded[:, 1:] |= white_mask[:, :-1]
        expanded[:, :-1] |= white_mask[:, 1:]
        white_mask = expanded
    image_data[white_mask] = white

    image = bpy.data.images.new(
        name="ControlledIdentityCoat",
        width=size,
        height=size,
        alpha=True,
    )
    image.colorspace_settings.name = "sRGB"
    image.pixels = image_data[::-1].reshape(-1).tolist()
    image.filepath_raw = str(texture_path)
    image.file_format = "PNG"
    image.save()
    png_codes = validate_coat_png_codes(
        texture_path.read_bytes(),
        plan,
        label="external coat",
    )
    image.pack()

    material = bpy.data.materials.new(name="ControlledIdentityCoat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    principled.inputs["Roughness"].default_value = 0.72
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(texture.outputs["Alpha"], principled.inputs["Alpha"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    mesh.materials.clear()
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = 0

    return {
        "resolution": [size, size],
        "face_role_counts": counts,
        "white_texel_count_after_bleed": int(np.count_nonzero(white_mask)),
        "white_texel_fraction_after_bleed": (
            float(np.count_nonzero(white_mask)) / float(size * size)
        ),
        "external_png_srgb_readback": png_codes,
    }


def canonical_skinned_surface_sha256(
    mesh_object,
    *,
    precision: int = POSITION_PRECISION,
    include_weights: bool = True,
) -> str:
    groups = mesh_object.vertex_groups
    mesh_object.data.calc_loop_triangles()
    triangles = []
    for triangle in mesh_object.data.loop_triangles:
        corners = []
        for index in triangle.vertices:
            vertex = mesh_object.data.vertices[index]
            corner = {
                "position": [round(float(value), precision) for value in vertex.co],
            }
            if include_weights:
                corner["weights"] = sorted(
                    [
                        [
                            groups[item.group].name,
                            round(float(item.weight), precision),
                        ]
                        for item in vertex.groups
                    ]
                )
            corners.append(corner)
        triangles.append(sorted(corners, key=contract.canonical_json))
    return hash_record(sorted(triangles, key=contract.canonical_json))


def canonical_surface_diagnostic(mesh_object) -> dict[str, dict[str, str]]:
    return {
        str(precision): {
            "positions_and_weights": canonical_skinned_surface_sha256(
                mesh_object,
                precision=precision,
            ),
            "positions_only": canonical_skinned_surface_sha256(
                mesh_object,
                precision=precision,
                include_weights=False,
            ),
        }
        for precision in range(3, 8)
    }


def glb_document_and_binary(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise RuntimeError("output GLB is truncated")
    magic, version, declared_size = struct.unpack_from("<III", payload, 0)
    if magic != GLB_MAGIC or version != 2 or declared_size != len(payload):
        raise RuntimeError("output GLB container header is invalid")
    json_payload = None
    binary_payload = None
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RuntimeError("output GLB chunk header is truncated")
        chunk_size, chunk_type = struct.unpack_from("<II", payload, offset)
        start = offset + 8
        end = start + chunk_size
        if end > len(payload):
            raise RuntimeError("output GLB chunk is truncated")
        if chunk_type == GLB_JSON_CHUNK:
            if json_payload is not None:
                raise RuntimeError("output GLB repeats its JSON chunk")
            json_payload = payload[start:end]
        elif chunk_type == GLB_BIN_CHUNK:
            if binary_payload is not None:
                raise RuntimeError("output GLB repeats its binary chunk")
            binary_payload = payload[start:end]
        offset = end
    if json_payload is None or binary_payload is None:
        raise RuntimeError("output GLB must contain JSON and binary chunks")
    try:
        document = contract.strict_json_loads(json_payload.rstrip(b" \t\r\n\x00"))
    except contract.IdentityContractError as error:
        raise RuntimeError(f"output GLB JSON is invalid: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError("output GLB JSON must be an object")
    return document, binary_payload


def glb_document(path: Path) -> dict[str, Any]:
    document, _binary = glb_document_and_binary(path)
    return document


def glb_float_scalar_accessor(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: Any,
) -> tuple[float, ...]:
    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])
    if not isinstance(accessor_index, int) or not 0 <= accessor_index < len(accessors):
        raise RuntimeError("animation input accessor is invalid")
    accessor = accessors[accessor_index]
    view_index = accessor.get("bufferView")
    count = accessor.get("count")
    if (
        accessor.get("componentType") != 5126
        or accessor.get("type") != "SCALAR"
        or accessor.get("sparse") is not None
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(buffer_views)
    ):
        raise RuntimeError("animation input accessor is not dense float SCALAR")
    view = buffer_views[view_index]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("animation input accessor uses an unsupported buffer")
    stride = view.get("byteStride", 4)
    if stride != 4:
        raise RuntimeError("animation input accessor is not tightly packed")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    end = start + count * stride
    if start < 0 or end > len(binary):
        raise RuntimeError("animation input accessor is out of bounds")
    return tuple(
        float(struct.unpack_from("<f", binary, start + index * stride)[0])
        for index in range(count)
    )


def raw_animation_time_signature(
    path: Path,
    plan: contract.IdentityPlan,
) -> dict[str, Any]:
    document, binary = glb_document_and_binary(path)
    animations = document.get("animations", [])
    records = {}
    for expected in plan.expected_actions:
        matches = [
            animation
            for animation in animations
            if isinstance(animation.get("name"), str)
            and expected.lower() in animation["name"].lower()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"GLB animation timing identity changed: {expected}")
        timelines = set()
        for sampler in matches[0].get("samplers", []):
            interpolation = sampler.get("interpolation", "LINEAR")
            if interpolation not in {"LINEAR", "STEP", "CUBICSPLINE"}:
                raise RuntimeError("GLB animation interpolation is invalid")
            times = glb_float_scalar_accessor(
                document,
                binary,
                sampler.get("input"),
            )
            if (
                tuple(sorted(times)) != times
                or len(set(times)) != len(times)
                or any(not math.isfinite(value) for value in times)
            ):
                raise RuntimeError("GLB animation timeline is not strictly increasing")
            timelines.add(
                (
                    interpolation,
                    tuple(rounded(value, 6) for value in times),
                )
            )
        if not timelines:
            raise RuntimeError(f"GLB animation has no timelines: {expected}")
        records[expected] = [
            {
                "interpolation": interpolation,
                "sample_count": len(times),
                "times_seconds": list(times),
            }
            for interpolation, times in sorted(timelines)
        ]
    fractional_24fps_times = sorted(
        {
            time_value
            for timelines in records.values()
            for timeline in timelines
            for time_value in timeline["times_seconds"]
            if abs(time_value * 24.0 - round(time_value * 24.0)) > 1.0e-5
        }
    )
    if not fractional_24fps_times:
        raise RuntimeError("animation timing evidence has no non-integer 24 fps times")
    return {
        "actions": records,
        "fractional_24fps_time_count": len(fractional_24fps_times),
        "preserves_non_integer_24fps_times": True,
        "sha256": hash_record(records),
    }


def matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            sum(
                float(left[row][index]) * float(right[index][column])
                for index in range(4)
            )
            for column in range(4)
        ]
        for row in range(4)
    ]


def gltf_node_local_matrix(node: Mapping[str, Any]) -> list[list[float]]:
    matrix = node.get("matrix")
    if matrix is not None:
        if (
            not isinstance(matrix, list)
            or len(matrix) != 16
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in matrix
            )
        ):
            raise RuntimeError("GLB node matrix is invalid")
        return [
            [float(matrix[column * 4 + row]) for column in range(4)] for row in range(4)
        ]
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or not isinstance(rotation, list)
        or len(rotation) != 4
        or not isinstance(scale, list)
        or len(scale) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in translation + rotation + scale
        )
    ):
        raise RuntimeError("GLB node TRS is invalid")
    x, y, z, w = (float(value) for value in rotation)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise RuntimeError("GLB node quaternion is degenerate")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    sx, sy, sz = (float(value) for value in scale)
    tx, ty, tz = (float(value) for value in translation)
    return [
        [
            (1.0 - 2.0 * (y * y + z * z)) * sx,
            (2.0 * (x * y - z * w)) * sy,
            (2.0 * (x * z + y * w)) * sz,
            tx,
        ],
        [
            (2.0 * (x * y + z * w)) * sx,
            (1.0 - 2.0 * (x * x + z * z)) * sy,
            (2.0 * (y * z - x * w)) * sz,
            ty,
        ],
        [
            (2.0 * (x * z - y * w)) * sx,
            (2.0 * (y * z + x * w)) * sy,
            (1.0 - 2.0 * (x * x + y * y)) * sz,
            tz,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]


def gltf_global_node_matrices(
    nodes: Sequence[Mapping[str, Any]],
    parent_by_node: Mapping[int, int],
) -> list[list[list[float]]]:
    resolved: dict[int, list[list[float]]] = {}
    resolving = set()

    def resolve(index: int) -> list[list[float]]:
        if index in resolved:
            return resolved[index]
        if index in resolving:
            raise RuntimeError("GLB node hierarchy contains a cycle")
        resolving.add(index)
        local = gltf_node_local_matrix(nodes[index])
        parent = parent_by_node.get(index)
        value = local if parent is None else matrix_multiply(resolve(parent), local)
        resolving.remove(index)
        resolved[index] = value
        return value

    return [resolve(index) for index in range(len(nodes))]


def raw_skin_signature(
    path: Path,
    *,
    precision: int,
) -> dict[str, Any]:
    document, binary = glb_document_and_binary(path)
    nodes = document.get("nodes", [])
    skins = document.get("skins", [])
    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])
    if len(skins) != 1 or not isinstance(nodes, list):
        raise RuntimeError("GLB skin signature requires one skin and node array")
    skin = skins[0]
    joints = skin.get("joints")
    accessor_index = skin.get("inverseBindMatrices")
    if (
        not isinstance(joints, list)
        or not joints
        or len(set(joints)) != len(joints)
        or any(
            not isinstance(index, int) or not 0 <= index < len(nodes)
            for index in joints
        )
        or not isinstance(accessor_index, int)
        or not 0 <= accessor_index < len(accessors)
    ):
        raise RuntimeError("GLB skin joint or inverse-bind accessor is invalid")
    parent_by_node: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        children = node.get("children", [])
        if not isinstance(children, list):
            raise RuntimeError("GLB node children must be an array")
        for child in children:
            if (
                not isinstance(child, int)
                or not 0 <= child < len(nodes)
                or child in parent_by_node
            ):
                raise RuntimeError("GLB joint hierarchy is ambiguous")
            parent_by_node[child] = parent_index
    joint_order = {node_index: index for index, node_index in enumerate(joints)}
    hierarchy = []
    for order_index, node_index in enumerate(joints):
        node = nodes[node_index]
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("GLB joint node is unnamed")
        parent_node = parent_by_node.get(node_index)
        hierarchy.append(
            {
                "joint_order": order_index,
                "name": name,
                "parent_joint_order": (
                    joint_order.get(parent_node) if parent_node is not None else None
                ),
            }
        )
    accessor = accessors[accessor_index]
    view_index = accessor.get("bufferView")
    if (
        accessor.get("componentType") != 5126
        or accessor.get("type") != "MAT4"
        or accessor.get("count") != len(joints)
        or accessor.get("sparse") is not None
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(buffer_views)
    ):
        raise RuntimeError("GLB inverse-bind accessor contract changed")
    view = buffer_views[view_index]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("GLB inverse-bind matrices use an unsupported buffer")
    stride = view.get("byteStride", 64)
    if stride != 64:
        raise RuntimeError("GLB inverse-bind matrix stride is not tightly packed")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    inverse_bind_matrices = []
    for index in range(len(joints)):
        matrix_offset = start + index * stride
        if matrix_offset < 0 or matrix_offset + 64 > len(binary):
            raise RuntimeError("GLB inverse-bind matrix is out of bounds")
        column_major = struct.unpack_from("<16f", binary, matrix_offset)
        inverse_bind_matrices.append(
            [
                [float(column_major[column * 4 + row]) for column in range(4)]
                for row in range(4)
            ]
        )
    global_matrices = gltf_global_node_matrices(nodes, parent_by_node)
    bind_products = [
        matrix_multiply(global_matrices[node_index], inverse_bind_matrices[index])
        for index, node_index in enumerate(joints)
    ]
    skinned_nodes = [
        index
        for index, node in enumerate(nodes)
        if node.get("skin") == 0 and isinstance(node.get("mesh"), int)
    ]
    if len(skinned_nodes) != 1:
        raise RuntimeError("GLB skin must be referenced by one mesh node")
    try:
        inverse_mesh_global = [
            [float(value) for value in row]
            for row in Matrix(global_matrices[skinned_nodes[0]]).inverted()
        ]
    except ValueError as error:
        raise RuntimeError("GLB skinned mesh global matrix is singular") from error
    bind_residuals = [
        matrix_multiply(inverse_mesh_global, matrix) for matrix in bind_products
    ]
    maximum_residual_magnitude = max(
        abs(float(matrix[row][column]) - (1.0 if row == column else 0.0))
        for matrix in bind_residuals
        for row in range(4)
        for column in range(4)
    )
    rounded_inverse_bind_matrices = [
        [rounded(value, precision) for row in matrix for value in row]
        for matrix in inverse_bind_matrices
    ]
    rounded_bind_residuals = [
        [rounded(value, precision) for row in matrix for value in row]
        for matrix in bind_residuals
    ]
    semantic_value = {
        "joint_hierarchy_and_order": hierarchy,
        "inverse_mesh_global_times_joint_global_times_inverse_bind": (
            rounded_bind_residuals
        ),
        "precision_decimals": precision,
    }
    return {
        "semantic_sha256": hash_record(semantic_value),
        "joint_count": len(joints),
        "joint_hierarchy_and_order_sha256": hash_record(hierarchy),
        "raw_inverse_bind_matrices_sha256": hash_record(rounded_inverse_bind_matrices),
        "bind_semantic_residual_sha256": hash_record(rounded_bind_residuals),
        "maximum_bind_residual_magnitude": maximum_residual_magnitude,
        "bind_semantic_residual_values": [
            [float(value) for row in matrix for value in row]
            for matrix in bind_residuals
        ],
        "precision_decimals": precision,
    }


def evaluated_world_surface_sha256(mesh_object, *, precision: int) -> str:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        evaluated_mesh.calc_loop_triangles()
        triangles = []
        for triangle in evaluated_mesh.loop_triangles:
            corners = [
                [
                    rounded(value, precision)
                    for value in (
                        evaluated.matrix_world @ evaluated_mesh.vertices[index].co
                    )
                ]
                for index in triangle.vertices
            ]
            triangles.append(sorted(corners, key=contract.canonical_json))
        return hash_record(sorted(triangles, key=contract.canonical_json))
    finally:
        evaluated.to_mesh_clear()


def evaluated_world_vertex_values(mesh_object) -> list[list[float]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        return [
            [float(value) for value in (evaluated.matrix_world @ vertex.co)]
            for vertex in evaluated_mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()


def bind_pose_world_surface_sha256(
    primary,
    armature,
    *,
    precision: int,
) -> str:
    previous_pose_position = armature.data.pose_position
    try:
        armature.data.pose_position = "REST"
        bpy.context.view_layer.update()
        return evaluated_world_surface_sha256(primary, precision=precision)
    finally:
        armature.data.pose_position = previous_pose_position
        bpy.context.view_layer.update()


def sampled_animation_semantics(
    primary,
    armature,
    plan: contract.IdentityPlan,
) -> dict[str, Any]:
    resolved = expected_actions(plan)
    scene = bpy.context.scene
    animation_data = armature.animation_data_create()
    previous_action = animation_data.action
    previous_frame = scene.frame_current
    tracks = list(animation_data.nla_tracks)
    previous_mutes = [track.mute for track in tracks]
    for track in tracks:
        track.mute = True
    root_bone = armature.pose.bones.get(plan.motion_root_bone)
    if root_bone is None or root_bone.parent is not None:
        raise RuntimeError("explicit motion root bone is missing or not a root")
    records = {}
    try:
        for expected in plan.expected_actions:
            action = bpy.data.actions.get(resolved[expected])
            if action is None:
                raise RuntimeError(f"animation action disappeared: {expected}")
            frame_start = float(action.frame_range[0])
            frame_end = float(action.frame_range[1])
            if not frame_end > frame_start:
                raise RuntimeError(f"animation action has no duration: {expected}")
            animation_data.action = action
            samples = []
            for phase in plan.roundtrip_animation_sample_phases:
                sample_frame = frame_start + phase * (frame_end - frame_start)
                integer_frame = math.floor(sample_frame)
                scene.frame_set(
                    integer_frame,
                    subframe=sample_frame - integer_frame,
                )
                bpy.context.view_layer.update()
                joint_world_matrices = [
                    {
                        "name": bone.name,
                        "matrix_world": [
                            rounded(
                                value,
                                plan.roundtrip_canonical_precision_decimals,
                            )
                            for row in (armature.matrix_world @ bone.matrix)
                            for value in row
                        ],
                    }
                    for bone in sorted(
                        armature.pose.bones,
                        key=lambda value: value.name,
                    )
                ]
                root_world_matrix = armature.matrix_world @ root_bone.matrix
                samples.append(
                    {
                        "phase": phase,
                        "joint_world_matrices_sha256": hash_record(
                            joint_world_matrices
                        ),
                        "skinned_world_surface_sha256": (
                            evaluated_world_surface_sha256(
                                primary,
                                precision=(plan.roundtrip_canonical_precision_decimals),
                            )
                        ),
                        "_skinned_world_vertex_values": (
                            evaluated_world_vertex_values(primary)
                        ),
                        "_root_world_translation": [
                            float(root_world_matrix[index][3]) for index in range(3)
                        ],
                    }
                )
            records[expected] = {
                "action_name": action.name,
                "duration_frames": frame_end - frame_start,
                "root_bone": root_bone.name,
                "samples": samples,
            }
    finally:
        animation_data.action = previous_action
        for track, mute in zip(tracks, previous_mutes):
            track.mute = mute
        scene.frame_set(previous_frame)
        bpy.context.view_layer.update()
    return {
        "precision_decimals": plan.roundtrip_canonical_precision_decimals,
        "sample_phases": list(plan.roundtrip_animation_sample_phases),
        "actions": records,
        "sha256": hash_record(records),
    }


def compare_sampled_animation_semantics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    maximum_delta_gate: float,
    maximum_duration_delta_gate: float,
) -> dict[str, Any]:
    import numpy as np

    if (
        before.get("precision_decimals") != after.get("precision_decimals")
        or before.get("sample_phases") != after.get("sample_phases")
        or set(before.get("actions", {})) != set(after.get("actions", {}))
    ):
        raise RuntimeError("sampled animation evidence structure changed")
    comparisons = {}
    overall_maximum_delta = 0.0
    overall_maximum_root_delta = 0.0
    overall_maximum_duration_delta = 0.0
    for action_name in sorted(before["actions"]):
        before_action = before["actions"][action_name]
        after_action = after["actions"][action_name]
        if (
            before_action.get("action_name") != after_action.get("action_name")
            or before_action.get("root_bone") != after_action.get("root_bone")
            or len(before_action.get("samples", []))
            != len(after_action.get("samples", []))
        ):
            raise RuntimeError("sampled animation action identity changed")
        duration_delta = abs(
            float(before_action["duration_frames"])
            - float(after_action["duration_frames"])
        )
        overall_maximum_duration_delta = max(
            overall_maximum_duration_delta,
            duration_delta,
        )
        if duration_delta > maximum_duration_delta_gate:
            raise RuntimeError(
                "animation duration exceeds the roundtrip delta gate: "
                f"action={action_name} observed={duration_delta} "
                f"gate={maximum_duration_delta_gate}"
            )
        action_comparisons = []
        before_root_origin = before_action["samples"][0]["_root_world_translation"]
        after_root_origin = after_action["samples"][0]["_root_world_translation"]
        for before_sample, after_sample in zip(
            before_action["samples"],
            after_action["samples"],
        ):
            before_values = before_sample.pop("_skinned_world_vertex_values")
            after_values = after_sample.pop("_skinned_world_vertex_values")
            before_root = before_sample.pop("_root_world_translation")
            after_root = after_sample.pop("_root_world_translation")
            if (
                before_sample.get("phase") != after_sample.get("phase")
                or not before_values
                or not after_values
                or any(len(vertex) != 3 for vertex in before_values + after_values)
            ):
                raise RuntimeError("sampled skinned vertex identity changed")

            def directed_maximum_nearest_distance(left, right):
                left_values = np.asarray(left, dtype=np.float64)
                right_values = np.asarray(right, dtype=np.float64)
                maximum_squared = 0.0
                for start in range(0, len(left_values), 128):
                    difference = (
                        left_values[start : start + 128, None, :]
                        - right_values[None, :, :]
                    )
                    nearest_squared = np.min(
                        np.sum(difference * difference, axis=2),
                        axis=1,
                    )
                    maximum_squared = max(
                        maximum_squared,
                        float(np.max(nearest_squared)),
                    )
                return math.sqrt(maximum_squared)

            maximum_delta = max(
                directed_maximum_nearest_distance(before_values, after_values),
                directed_maximum_nearest_distance(after_values, before_values),
            )
            maximum_root_delta = max(
                abs(
                    (float(before_root[index]) - float(before_root_origin[index]))
                    - (float(after_root[index]) - float(after_root_origin[index]))
                )
                for index in range(3)
            )
            overall_maximum_delta = max(overall_maximum_delta, maximum_delta)
            overall_maximum_root_delta = max(
                overall_maximum_root_delta,
                maximum_root_delta,
            )
            action_comparisons.append(
                {
                    "phase": before_sample["phase"],
                    "source_evaluated_vertex_count": len(before_values),
                    "output_evaluated_vertex_count": len(after_values),
                    "maximum_skinned_world_vertex_delta": maximum_delta,
                    "within_gate": maximum_delta <= maximum_delta_gate,
                    "maximum_root_trajectory_delta": maximum_root_delta,
                    "root_trajectory_within_gate": (
                        maximum_root_delta <= maximum_delta_gate
                    ),
                }
            )
        comparisons[action_name] = action_comparisons
    if (
        overall_maximum_delta > maximum_delta_gate
        or overall_maximum_root_delta > maximum_delta_gate
    ):
        raise RuntimeError(
            "sampled skinned world motion exceeds the roundtrip delta gate: "
            f"observed={overall_maximum_delta} gate={maximum_delta_gate}"
        )
    return {
        "maximum_skinned_world_vertex_delta": overall_maximum_delta,
        "maximum_skinned_world_vertex_delta_gate": maximum_delta_gate,
        "maximum_root_trajectory_delta": overall_maximum_root_delta,
        "maximum_root_trajectory_delta_gate": maximum_delta_gate,
        "maximum_animation_duration_delta_frames": (overall_maximum_duration_delta),
        "maximum_animation_duration_delta_frames_gate": (maximum_duration_delta_gate),
        "per_action": comparisons,
        "joint_world_matrix_hashes_recorded_but_not_used_as_blender_rig_gate": True,
        "overall": "passed",
    }


def raw_glb_readback(path: Path, plan: contract.IdentityPlan) -> dict[str, Any]:
    document, binary = glb_document_and_binary(path)
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    materials = document.get("materials", [])
    images = document.get("images", [])
    textures = document.get("textures", [])
    nodes = document.get("nodes", [])
    if (
        len(meshes) != 1
        or len(skins) != 1
        or len(animations) != len(plan.expected_actions)
        or len(materials) != 1
        or len(images) != 1
        or len(textures) != 1
        or any(node.get("name") == plan.removal_object for node in nodes)
    ):
        raise RuntimeError("output GLB inventory contradicts the identity plan")
    action_names = [animation.get("name") for animation in animations]
    for expected in plan.expected_actions:
        if (
            sum(
                isinstance(name, str) and expected.lower() in name.lower()
                for name in action_names
            )
            != 1
        ):
            raise RuntimeError(f"output GLB lost action {expected}")
    channel_counts = {
        str(animation.get("name")): len(animation.get("channels", []))
        for animation in animations
    }
    primitives = [
        primitive for mesh in meshes for primitive in mesh.get("primitives", [])
    ]
    if not primitives:
        raise RuntimeError("output GLB has no mesh primitives")
    required_attributes = {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    for primitive in primitives:
        if not required_attributes.issubset(
            primitive.get("attributes", {})
        ) or not isinstance(primitive.get("material"), int):
            raise RuntimeError("output GLB primitive lost UV, skin, or material data")
    image = images[0]
    if (
        not isinstance(image.get("bufferView"), int)
        or image.get("mimeType") != "image/png"
        or "uri" in image
    ):
        raise RuntimeError("output coat texture is not embedded PNG evidence")
    image_view = document.get("bufferViews", [])[image["bufferView"]]
    image_start = int(image_view.get("byteOffset", 0))
    image_end = image_start + int(image_view.get("byteLength", 0))
    if (
        image_view.get("buffer", 0) != 0
        or image_start < 0
        or image_end <= image_start
        or image_end > len(binary)
    ):
        raise RuntimeError("embedded coat PNG buffer view is invalid")
    embedded_png_codes = validate_coat_png_codes(
        binary[image_start:image_end],
        plan,
        label="embedded GLB coat",
    )
    pbr = materials[0].get("pbrMetallicRoughness", {})
    if not isinstance(pbr.get("baseColorTexture", {}).get("index"), int):
        raise RuntimeError("output material does not reference the embedded coat")
    return {
        "mesh_count": len(meshes),
        "skin_count": len(skins),
        "animation_names": action_names,
        "animation_channel_counts": channel_counts,
        "material_count": len(materials),
        "texture_count": len(textures),
        "image_count": len(images),
        "embedded_png": True,
        "embedded_png_srgb_readback": embedded_png_codes,
        "base_color_texture_color_space": "sRGB",
        "removal_object_absent": True,
        "primitive_count": len(primitives),
        "required_attributes_present": True,
    }


def apply_ear_morph(primary, plan: contract.IdentityPlan) -> dict[str, Any]:
    mesh = primary.data
    selected = set(plan.positive.indices) | set(plan.negative.indices)
    unselected = [
        vertex.index for vertex in mesh.vertices if vertex.index not in selected
    ]
    before_unselected = vertex_position_sha256(mesh, unselected)
    before_selected = vertex_position_sha256(mesh, sorted(selected))
    maximum_displacement = 0.0
    for side in (plan.positive, plan.negative):
        for index in side.indices:
            vertex = mesh.vertices[index]
            source = tuple(float(value) for value in vertex.co)
            output = contract.rotate_ear_point(source, side)
            maximum_displacement = max(
                maximum_displacement,
                math.dist(source, output),
            )
            vertex.co = output
    after_unselected = vertex_position_sha256(mesh, unselected)
    after_selected = vertex_position_sha256(mesh, sorted(selected))
    if before_unselected != after_unselected or before_selected == after_selected:
        raise RuntimeError("ear morph escaped its explicit vertex masks")
    _minimum, _maximum, diagonal = bbox(mesh)
    if maximum_displacement / diagonal > plan.maximum_displacement_diagonal_ratio:
        raise RuntimeError("realized ear displacement exceeds the plan gate")
    positive = mask_groups(mesh, plan.positive.indices)
    negative = mask_groups(mesh, plan.negative.indices)
    pair_mirrored_groups(
        positive,
        negative,
        plane_y=plan.symmetry_plane_y,
        tolerance=plan.symmetry_tolerance * 2.0,
    )
    return {
        "moved_vertex_count": len(selected),
        "maximum_displacement": maximum_displacement,
        "maximum_displacement_bbox_diagonal_ratio": (maximum_displacement / diagonal),
        "selected_position_sha256_before": before_selected,
        "selected_position_sha256_after": after_selected,
        "unselected_position_sha256_before": before_unselected,
        "unselected_position_sha256_after": after_unselected,
        "unselected_positions_unchanged": True,
        "output_masks_mirrored": True,
    }


def export_glb(primary, armature, output_path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    primary.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_extra_animations=True,
        export_animation_mode="ACTIONS",
        # Imported glTF actions retain exact fractional Blender-frame key times
        # (for example 0.8-frame spacing for a 30 Hz clip in a 24 fps scene).
        # Forced sampling rounds that domain to integer export frames and changes
        # the continuous animation between keys.  Export the original key times.
        export_force_sampling=False,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    if "FINISHED" not in result or not output_path.is_file():
        raise RuntimeError("Blender GLB export did not finish")


def realize(
    *,
    plan: contract.IdentityPlan,
    preflight_path: Path,
    preflight_payload: bytes,
    preflight_snapshot_path: Path,
    authorization_path: Path,
    authorization_payload: bytes,
    authorization_snapshot_path: Path,
    source_path: Path,
    plan_path: Path,
    output_glb: Path,
    texture_path: Path,
    toolchain_snapshots: Mapping[str, Any],
    source_snapshot_path: Path,
    plan_snapshot_path: Path,
) -> dict[str, Any]:
    primary, armature, removal = resolve_scene(plan)
    before = {
        "inventory": mesh_inventory(),
        "topology_sha256": mesh_topology_sha256(primary.data),
        "weights_sha256": vertex_weight_sha256(primary),
        "skeleton_sha256": skeleton_sha256(armature),
        "actions_sha256": action_sha256(),
        "canonical_skinned_surface_sha256": canonical_skinned_surface_sha256(primary),
    }
    morph = apply_ear_morph(primary, plan)
    removed_custom_shape_assignments = sorted(
        bone.name for bone in armature.pose.bones if bone.custom_shape is removal
    )
    if len(removed_custom_shape_assignments) != plan.removal_custom_shape_assignments:
        raise RuntimeError("pose-bone custom-shape assignment count changed")
    for bone in armature.pose.bones:
        if bone.custom_shape is removal:
            bone.custom_shape = None
    if any(bone.custom_shape is not None for bone in armature.pose.bones):
        raise RuntimeError("pose-bone custom-shape references survived removal")
    removal_mesh = removal.data
    bpy.data.objects.remove(removal, do_unlink=True)
    if bpy.data.objects.get(plan.removal_object) is not None:
        raise RuntimeError("unskinned removal target survived deletion")
    if removal_mesh.users != 0:
        raise RuntimeError("unskinned removal mesh datablock still has users")
    bpy.data.meshes.remove(removal_mesh)
    if bpy.data.meshes.get(plan.removal_object) is not None:
        raise RuntimeError("unskinned removal mesh datablock survived deletion")
    del removal_mesh
    del removal
    generate_uv(primary, plan)
    texture = rasterize_coat_texture(primary, plan, texture_path)
    after_memory = {
        "inventory": mesh_inventory(),
        "topology_sha256": mesh_topology_sha256(primary.data),
        "weights_sha256": vertex_weight_sha256(primary),
        "skeleton_sha256": skeleton_sha256(armature),
        "actions_sha256": action_sha256(),
        "uv_layer_names": [layer.name for layer in primary.data.uv_layers],
        "uv_loop_count": len(primary.data.uv_layers.active.data),
        "canonical_skinned_surface_sha256": canonical_skinned_surface_sha256(primary),
    }
    for key in (
        "topology_sha256",
        "weights_sha256",
        "skeleton_sha256",
        "actions_sha256",
    ):
        if before[key] != after_memory[key]:
            raise RuntimeError(f"in-memory invariant changed unexpectedly: {key}")
    if len(after_memory["inventory"]) != len(before["inventory"]) - 1:
        raise RuntimeError("scene inventory changed by more than the removal target")
    surface_before_export = canonical_skinned_surface_sha256(
        primary,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    skeleton_hierarchy_before_export = skeleton_hierarchy_sha256(armature)
    bind_world_surface_before_export = bind_pose_world_surface_sha256(
        primary,
        armature,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    animation_semantics_before_export = sampled_animation_semantics(
        primary,
        armature,
        plan,
    )
    surface_diagnostic_before = canonical_surface_diagnostic(primary)
    export_glb(primary, armature, output_glb)
    raw_readback = raw_glb_readback(output_glb, plan)
    source_animation_time_signature = raw_animation_time_signature(
        source_snapshot_path,
        plan,
    )
    output_animation_time_signature = raw_animation_time_signature(output_glb, plan)
    if (
        source_animation_time_signature["actions"]
        != output_animation_time_signature["actions"]
    ):
        raise RuntimeError(
            "output GLB changed raw animation timelines or interpolation"
        )
    source_skin_signature = raw_skin_signature(
        source_snapshot_path,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    output_skin_signature = raw_skin_signature(
        output_glb,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    source_bind_residuals = source_skin_signature.pop("bind_semantic_residual_values")
    output_bind_residuals = output_skin_signature.pop("bind_semantic_residual_values")
    if len(source_bind_residuals) != len(output_bind_residuals):
        raise RuntimeError("output GLB changed bind residual matrix count")
    maximum_bind_semantic_delta = max(
        abs(float(source) - float(output))
        for source_matrix, output_matrix in zip(
            source_bind_residuals,
            output_bind_residuals,
        )
        for source, output in zip(source_matrix, output_matrix)
    )
    if (
        source_skin_signature["joint_hierarchy_and_order_sha256"]
        != output_skin_signature["joint_hierarchy_and_order_sha256"]
        or maximum_bind_semantic_delta > plan.maximum_bind_matrix_semantic_delta
    ):
        raise RuntimeError(
            "output GLB changed joint order or bind-matrix semantics: "
            + json.dumps(
                {
                    "source": source_skin_signature,
                    "output": output_skin_signature,
                },
                sort_keys=True,
            )
        )

    import_snapshot(output_glb, disable_bone_shape=True)
    output_primary = bpy.data.objects.get(plan.primary_mesh)
    output_armature = bpy.data.objects.get(plan.armature)
    output_inventory_diagnostic = {
        "objects": [
            {
                "name": item.name,
                "type": item.type,
                "data_name": item.data.name if item.data is not None else None,
            }
            for item in sorted(bpy.data.objects, key=lambda value: value.name)
        ],
        "images": [
            item.name for item in sorted(bpy.data.images, key=lambda value: value.name)
        ],
        "primary_uv_layers": (
            [layer.name for layer in output_primary.data.uv_layers]
            if output_primary is not None and output_primary.type == "MESH"
            else None
        ),
    }
    if (
        output_primary is None
        or output_primary.type != "MESH"
        or output_armature is None
        or output_armature.type != "ARMATURE"
        or any(
            item.type == "MESH" and item.name != plan.primary_mesh
            for item in bpy.data.objects
        )
        or len(output_primary.data.uv_layers) != 1
        or len(bpy.data.images) != 1
    ):
        raise RuntimeError(
            "Blender output reimport inventory is invalid: "
            + json.dumps(output_inventory_diagnostic, sort_keys=True)
        )
    expected_actions(plan)
    surface_after_reimport = canonical_skinned_surface_sha256(
        output_primary,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    surface_diagnostic_after = canonical_surface_diagnostic(output_primary)
    if surface_before_export != surface_after_reimport:
        raise RuntimeError(
            "output reimport changed canonical geometry or skin weights: "
            + json.dumps(
                {
                    "before": surface_diagnostic_before,
                    "after": surface_diagnostic_after,
                },
                sort_keys=True,
            )
        )
    skeleton_hierarchy_after_reimport = skeleton_hierarchy_sha256(output_armature)
    if skeleton_hierarchy_after_reimport != skeleton_hierarchy_before_export:
        raise RuntimeError("output reimport changed the skeleton hierarchy")
    bind_world_surface_after_reimport = bind_pose_world_surface_sha256(
        output_primary,
        output_armature,
        precision=plan.roundtrip_canonical_precision_decimals,
    )
    if bind_world_surface_after_reimport != bind_world_surface_before_export:
        raise RuntimeError("output reimport changed REST skinned world positions")
    animation_semantics_after_reimport = sampled_animation_semantics(
        output_primary,
        output_armature,
        plan,
    )
    animation_semantic_comparison = compare_sampled_animation_semantics(
        animation_semantics_before_export,
        animation_semantics_after_reimport,
        maximum_delta_gate=plan.maximum_sampled_skinned_world_delta,
        maximum_duration_delta_gate=(plan.maximum_animation_duration_delta_frames),
    )

    return {
        "schema": contract.REALIZATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_candidate_pending_final_animation_review",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "lineage": {
            "source": {
                "path": str(source_path),
                "sha256": sha256_file(source_snapshot_path),
                "size_bytes": source_snapshot_path.stat().st_size,
                "publication_snapshot": evidence_file_record(source_snapshot_path),
            },
            "plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_snapshot_path),
                "size_bytes": plan_snapshot_path.stat().st_size,
                "publication_snapshot": evidence_file_record(plan_snapshot_path),
            },
            "preflight": {
                "path": str(preflight_path),
                "sha256": sha256_bytes(preflight_payload),
                "size_bytes": len(preflight_payload),
                "publication_snapshot": evidence_file_record(preflight_snapshot_path),
            },
            "machine_execution_authorization": {
                "path": str(authorization_path),
                "sha256": sha256_bytes(authorization_payload),
                "size_bytes": len(authorization_payload),
                "publication_snapshot": evidence_file_record(
                    authorization_snapshot_path
                ),
            },
        },
        "toolchain": toolchain_record(),
        "toolchain_publication_snapshots": toolchain_snapshots,
        "publication_security_boundary": contract.publication_security_boundary(),
        "operations": {
            "ear_morph": morph,
            "removed_object": plan.removal_object,
            "removed_pose_bone_custom_shape_assignments": (
                removed_custom_shape_assignments
            ),
            "deterministic_uv": {
                "method": "blender_smart_project_v1",
                "angle_limit_radians": plan.uv_angle_limit,
                "island_margin": plan.uv_island_margin,
            },
            "deterministic_coat": texture,
        },
        "invariants": {
            "before": before,
            "after_in_memory": after_memory,
            "topology_unchanged_in_memory": True,
            "weights_unchanged_in_memory": True,
            "skeleton_unchanged_in_memory": True,
            "actions_unchanged_in_memory": True,
            "canonical_skinned_surface_sha256_before_export": (surface_before_export),
            "canonical_skinned_surface_sha256_after_reimport": (surface_after_reimport),
            "canonical_roundtrip_precision_decimals": (
                plan.roundtrip_canonical_precision_decimals
            ),
            "canonical_geometry_and_weights_survived_reimport": True,
            "canonical_triangle_surface_and_weight_clusters_survived_reimport": (True),
            "skeleton_hierarchy_sha256_before_export": (
                skeleton_hierarchy_before_export
            ),
            "skeleton_hierarchy_sha256_after_reimport": (
                skeleton_hierarchy_after_reimport
            ),
            "raw_source_skin_signature": source_skin_signature,
            "raw_output_skin_signature": output_skin_signature,
            "maximum_bind_matrix_semantic_delta": maximum_bind_semantic_delta,
            "maximum_bind_matrix_semantic_delta_gate": (
                plan.maximum_bind_matrix_semantic_delta
            ),
            "bind_world_surface_sha256_before_export": (
                bind_world_surface_before_export
            ),
            "bind_world_surface_sha256_after_reimport": (
                bind_world_surface_after_reimport
            ),
            "skeleton_survived_reimport": True,
            "animation_semantics_before_export": (animation_semantics_before_export),
            "animation_semantics_after_reimport": (animation_semantics_after_reimport),
            "animation_semantic_comparison": animation_semantic_comparison,
            "raw_source_animation_time_signature": (source_animation_time_signature),
            "raw_output_animation_time_signature": (output_animation_time_signature),
            "raw_animation_timelines_and_interpolation_unchanged": True,
            "idle_and_walking_actions_survived_reimport": True,
            "canonical_surface_precision_diagnostic": surface_diagnostic_after,
        },
        "output_readback": raw_readback,
        "blender_reimport": {
            "disable_bone_shape": True,
            "synthetic_bone_shape_object_absent": True,
        },
        "outputs": {
            "glb": relative_file_record(output_glb),
            "coat_texture": relative_file_record(texture_path),
        },
        "authority_boundary": {
            "owner_animation_decision": "pending",
            "breed_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "dynamic_audio": "pending",
            "native_merge_authorized": False,
        },
        "next_gate": "final_six_view_animation_and_identity_review",
    }


def main() -> int:
    args = parse_argv()
    source_path, source_payload = authenticated_bytes(
        args.input,
        args.input_sha256,
        "source GLB",
    )
    plan_path, plan_payload = authenticated_bytes(
        args.plan,
        args.plan_sha256,
        "identity plan",
    )
    plan_value = contract.strict_json_loads(plan_payload)
    plan = contract.load_plan(plan_value)
    if plan.source_sha256 != sha256_bytes(
        source_payload
    ) or plan.source_size_bytes != len(source_payload):
        raise RuntimeError("identity plan source binding changed")

    publication = contract.open_secure_publication(args.output_root)
    publication_closed = False
    try:
        source_snapshot_path = publication.write_evidence(
            "source_snapshot.glb",
            source_payload,
        )
        plan_snapshot_path = publication.write_evidence(
            "identity_plan.json",
            plan_payload,
        )
        toolchain_snapshots = snapshot_toolchain(publication)
        import_snapshot(source_snapshot_path)
        primary, armature, removal = resolve_scene(plan)
        core = audit_core(plan, primary, armature, removal)
        receipt = audit_receipt(
            source_path=source_path,
            source_payload=source_payload,
            plan_path=plan_path,
            plan_payload=plan_payload,
            plan=plan,
            core=core,
            toolchain_snapshots=toolchain_snapshots,
            source_snapshot_path=source_snapshot_path,
            plan_snapshot_path=plan_snapshot_path,
        )
        if args.mode == "audit":
            publication.write_root_json("preflight_receipt.json", receipt)
            publication_records = publication.seal(
                expected_root_files={"preflight_receipt.json"},
                expected_evidence_files={
                    "source_snapshot.glb",
                    "identity_plan.json",
                    "blender_realize_bounded_quadruped_identity.py",
                    "bounded_quadruped_identity_contract.py",
                },
            )
            require_snapshot_publication_records(
                publication_records,
                {
                    "evidence/source_snapshot.glb": {
                        "path": "evidence/source_snapshot.glb",
                        "sha256": sha256_bytes(source_payload),
                        "size_bytes": len(source_payload),
                    },
                    "evidence/identity_plan.json": {
                        "path": "evidence/identity_plan.json",
                        "sha256": sha256_bytes(plan_payload),
                        "size_bytes": len(plan_payload),
                    },
                    **{
                        record["path"]: record
                        for record in toolchain_snapshots.values()
                    },
                },
            )
            publication.publish(publication_records)
            publication.close()
            publication_closed = True
            print(
                "BOUNDED_QUADRUPED_IDENTITY_PREFLIGHT_OK "
                f"output={publication.output_root}",
                flush=True,
            )
            return 0

        preflight_path, preflight_payload = authenticated_bytes(
            args.preflight,
            args.preflight_sha256,
            "preflight receipt",
        )
        preflight = contract.strict_json_loads(preflight_payload)
        if (
            not isinstance(preflight, Mapping)
            or preflight.get("schema") != contract.PREFLIGHT_SCHEMA
            or preflight.get("status") != "pending_machine_execution_authorization"
            or preflight.get("formal_dataset_registration_authorized") is not False
            or preflight.get("execution_authorized") is not False
            or preflight.get("source", {}).get("sha256") != sha256_bytes(source_payload)
            or preflight.get("plan", {}).get("sha256") != sha256_bytes(plan_payload)
            or preflight.get("audit_core_sha256") != contract.canonical_sha256(core)
            or preflight.get("toolchain") != toolchain_record()
        ):
            raise RuntimeError("preflight receipt no longer matches the fresh audit")
        authorization_path, authorization_payload = authenticated_bytes(
            args.execution_authorization,
            args.execution_authorization_sha256,
            "machine execution authorization",
        )
        authorization = contract.strict_json_loads(authorization_payload)
        contract.validate_execution_authorization(
            authorization,
            expected_preflight_sha256=sha256_bytes(preflight_payload),
            expected_plan_sha256=sha256_bytes(plan_payload),
            expected_source_sha256=sha256_bytes(source_payload),
        )
        preflight_snapshot_path = publication.write_evidence(
            "preflight_receipt.json",
            preflight_payload,
        )
        authorization_snapshot_path = publication.write_evidence(
            "machine_execution_authorization.json",
            authorization_payload,
        )

        output_glb = publication.staging_path / "bounded_identity.glb"
        texture_path = publication.staging_path / "identity_coat.png"
        manifest = realize(
            plan=plan,
            preflight_path=preflight_path,
            preflight_payload=preflight_payload,
            preflight_snapshot_path=preflight_snapshot_path,
            authorization_path=authorization_path,
            authorization_payload=authorization_payload,
            authorization_snapshot_path=authorization_snapshot_path,
            source_path=source_path,
            plan_path=plan_path,
            output_glb=output_glb,
            texture_path=texture_path,
            toolchain_snapshots=toolchain_snapshots,
            source_snapshot_path=source_snapshot_path,
            plan_snapshot_path=plan_snapshot_path,
        )
        publication.register_root_file("bounded_identity.glb")
        publication.register_root_file("identity_coat.png")
        publication.write_root_json("realization_manifest.json", manifest)
        publication_records = publication.seal(
            expected_root_files={
                "bounded_identity.glb",
                "identity_coat.png",
                "realization_manifest.json",
            },
            expected_evidence_files={
                "source_snapshot.glb",
                "identity_plan.json",
                "blender_realize_bounded_quadruped_identity.py",
                "bounded_quadruped_identity_contract.py",
                "preflight_receipt.json",
                "machine_execution_authorization.json",
            },
        )
        require_snapshot_publication_records(
            publication_records,
            {
                "evidence/source_snapshot.glb": {
                    "path": "evidence/source_snapshot.glb",
                    "sha256": sha256_bytes(source_payload),
                    "size_bytes": len(source_payload),
                },
                "evidence/identity_plan.json": {
                    "path": "evidence/identity_plan.json",
                    "sha256": sha256_bytes(plan_payload),
                    "size_bytes": len(plan_payload),
                },
                "evidence/preflight_receipt.json": {
                    "path": "evidence/preflight_receipt.json",
                    "sha256": sha256_bytes(preflight_payload),
                    "size_bytes": len(preflight_payload),
                },
                "evidence/machine_execution_authorization.json": {
                    "path": "evidence/machine_execution_authorization.json",
                    "sha256": sha256_bytes(authorization_payload),
                    "size_bytes": len(authorization_payload),
                },
                **{record["path"]: record for record in toolchain_snapshots.values()},
            },
        )
        for output_name, manifest_record in (
            ("bounded_identity.glb", manifest["outputs"]["glb"]),
            ("identity_coat.png", manifest["outputs"]["coat_texture"]),
        ):
            if publication_records[output_name] != manifest_record:
                raise RuntimeError(
                    f"sealed output no longer matches manifest: {output_name}"
                )
        publication.publish(publication_records)
        publication.close()
        publication_closed = True
        print(
            "BOUNDED_QUADRUPED_IDENTITY_REALIZATION_OK "
            f"output={publication.output_root}",
            flush=True,
        )
        return 0
    except Exception as error:
        if publication.published:
            raise
        try:
            publication.cleanup()
        except contract.IdentityContractError as cleanup_error:
            raise RuntimeError(
                f"{error}; staging quarantine/retention status: {cleanup_error}"
            ) from error
        raise
    finally:
        if not publication_closed:
            publication.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        contract.IdentityContractError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"BOUNDED_QUADRUPED_IDENTITY_FAILED {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
