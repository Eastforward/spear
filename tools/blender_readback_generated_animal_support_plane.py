"""Independently re-import and verify a generated-animal leveling export.

The leveling producer measures an in-memory Blender scene before exporting a
GLB.  That scene is not evidence that the bytes written to disk retained the
same rigid transform, skeleton, topology, or post-TokenRig skin weights.  This
separate Blender process imports both the authenticated pre-level GLB and the
exported GLB, recomputes quadruped semantics and both visible-foot authorities,
and proves that exactly one rigid transform relates the two artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_animal_support_plane import (  # noqa: E402
    evaluate_dual_authority_support_plane,
)
from tools.generated_animal_support_plane_contract import (  # noqa: E402
    MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG,
    MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA,
    MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO,
    MAXIMUM_RIGID_VERTEX_DELTA_RATIO,
    MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO,
    MAXIMUM_SKIN_WEIGHT_DELTA,
    OUTPUT_READBACK_SCHEMA,
    validate_rigid_leveling_transform,
)
from tools.generated_quadruped_semantics import (  # noqa: E402
    infer_quadruped_semantics,
)


SCHEMA = OUTPUT_READBACK_SCHEMA


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-level-input", type=Path, required=True)
    parser.add_argument("--leveled-output", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--rig-audit", type=Path, required=True)
    parser.add_argument(
        "--front-axis",
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        required=True,
    )
    parser.add_argument("--readback-output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_new_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace support-plane readback: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def path_hash_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def matrix_record(matrix) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def scene_summary() -> dict:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ]
    skinned = [
        obj
        for obj in meshes
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    return {
        "mesh_count": len(meshes),
        "skinned_mesh_count": len(skinned),
        "armature_count": len(armatures),
        "bone_count": sum(len(obj.data.bones) for obj in armatures),
        "material_count": len(bpy.data.materials),
        "image_count": len(bpy.data.images),
        "action_count": len(bpy.data.actions),
    }


def semantic_records(armature) -> list[dict]:
    records = []
    for bone in armature.data.bones:
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        records.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "children": sorted(child.name for child in bone.children),
                "head_world": [float(value) for value in head],
                "tail_world": [float(value) for value in tail],
            }
        )
    return sorted(records, key=lambda record: record["name"])


def world_vertices(mesh) -> np.ndarray:
    local = np.empty((len(mesh.data.vertices), 3), dtype=np.float64)
    mesh.data.vertices.foreach_get("co", local.ravel())
    matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def mesh_bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return minimum, maximum - minimum


def topology_sha256(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(f"vertices:{len(mesh.data.vertices)}\n".encode())
    for polygon in mesh.data.polygons:
        digest.update(
            (
                f"{polygon.index}:"
                + ",".join(str(index) for index in polygon.vertices)
                + "\n"
            ).encode()
        )
    return digest.hexdigest()


def weight_matrix(mesh, bone_names: list[str]) -> np.ndarray:
    bone_index = {name: index for index, name in enumerate(bone_names)}
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    weights = np.zeros(
        (len(mesh.data.vertices), len(bone_names)), dtype=np.float64
    )
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            column = bone_index.get(group_names.get(membership.group))
            if column is not None:
                weights[vertex.index, column] += float(membership.weight)
    return weights


def distal_owner_bones(semantics) -> list[list[str]]:
    chain_by_leaf = {}
    for label in (
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    ):
        chain = list(getattr(semantics, label))
        if not chain or chain[-1] not in semantics.foot_leaves:
            raise RuntimeError(f"incomplete semantic limb chain: {label}")
        chain_by_leaf[chain[-1]] = chain
    if set(chain_by_leaf) != set(semantics.foot_leaves):
        raise RuntimeError("semantic limb chains do not cover all four feet")
    result = []
    seen = set()
    for leaf in semantics.foot_leaves:
        chain = chain_by_leaf[leaf]
        if len(chain) < 2:
            raise RuntimeError(f"semantic foot chain is too short: {leaf}")
        owners = chain[-2:]
        if seen.intersection(owners):
            raise RuntimeError("distal weight-owner bones overlap between feet")
        seen.update(owners)
        result.append(owners)
    return result


def distal_weight_scores(
    mesh, foot_leaves: tuple[str, ...], owner_bones: list[list[str]]
) -> np.ndarray:
    owner_by_bone = {
        bone_name: foot_index
        for foot_index, names in enumerate(owner_bones)
        for bone_name in names
    }
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    if not set(owner_by_bone).issubset(set(group_names.values())):
        raise RuntimeError(
            "output GLB is missing a distal semantic weight-owner group"
        )
    if len(foot_leaves) != 4:
        raise RuntimeError("support-plane readback needs exactly four feet")
    scores = np.zeros((len(mesh.data.vertices), 4), dtype=np.float64)
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            owner = owner_by_bone.get(group_names.get(membership.group))
            if owner is not None:
                scores[vertex.index, owner] += float(membership.weight)
    return scores


def snapshot(path: Path, front_axis: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ]
    skinned = [
        obj
        for obj in meshes
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    summary = scene_summary()
    if len(skinned) != 1 or len(armatures) != 1:
        raise RuntimeError(
            f"readback expected one skinned mesh and one armature: {summary}"
        )
    if summary["action_count"] != 0:
        raise RuntimeError("support-plane output unexpectedly contains animation")
    mesh = skinned[0]
    armature = armatures[0]
    vertices = world_vertices(mesh)
    minimum, extent = mesh_bbox(vertices)
    records = semantic_records(armature)
    semantics = infer_quadruped_semantics(
        records,
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=front_axis,
    )
    by_name = {record["name"]: record for record in records}
    owners = distal_owner_bones(semantics)
    segment_heads = np.asarray(
        [by_name[name]["head_world"] for name in semantics.foot_leaves],
        dtype=np.float64,
    )
    segment_tails = np.asarray(
        [by_name[name]["tail_world"] for name in semantics.foot_leaves],
        dtype=np.float64,
    )
    scores = distal_weight_scores(mesh, semantics.foot_leaves, owners)
    diagonal = float(np.linalg.norm(extent))
    dual = evaluate_dual_authority_support_plane(
        vertices,
        segment_heads,
        segment_tails,
        scores,
        mesh_diagonal=diagonal,
        maximum_residual_ratio=0.02,
        maximum_tilt_deg=30.0,
    )
    limb_chains = {
        label: list(getattr(semantics, label))
        for label in (
            "front_side_negative",
            "front_side_positive",
            "hind_side_negative",
            "hind_side_positive",
        )
    }
    bone_names = sorted(record["name"] for record in records)
    return {
        "scene": summary,
        "mesh_name": mesh.name,
        "armature_name": armature.name,
        "mesh_world_matrix": matrix_record(mesh.matrix_world),
        "armature_world_matrix": matrix_record(armature.matrix_world),
        "root_objects": [
            {
                "name": obj.name,
                "type": obj.type,
                "world_matrix": matrix_record(obj.matrix_world),
            }
            for obj in sorted(
                (obj for obj in bpy.context.scene.objects if obj.parent is None),
                key=lambda obj: (obj.type, obj.name),
            )
        ],
        "vertices": vertices,
        "mesh_bbox_min": minimum,
        "mesh_bbox_extent": extent,
        "mesh_diagonal": diagonal,
        "polygon_count": len(mesh.data.polygons),
        "topology_sha256": topology_sha256(mesh),
        "materials": [slot.material.name for slot in mesh.material_slots],
        "uv_layers": [layer.name for layer in mesh.data.uv_layers],
        "records": records,
        "bone_names": bone_names,
        "weights": weight_matrix(mesh, bone_names),
        "semantic_rig": {
            "foot_leaves": list(semantics.foot_leaves),
            "limb_chains": limb_chains,
            "distal_owner_bones": owners,
        },
        "dual_authority": dual,
    }


def transform_points(points: np.ndarray, rotation, translation: float):
    result = points @ np.asarray(rotation, dtype=np.float64).T
    result[:, 2] += float(translation)
    return result


def maximum_point_delta(actual, expected, label: str) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise RuntimeError(f"{label} point shape changed: {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise RuntimeError(f"{label} contains non-finite points")
    return float(np.linalg.norm(actual - expected, axis=1).max())


def _grid_key(point, cell_size: float) -> tuple[int, int, int]:
    return tuple(int(math.floor(float(value) / cell_size)) for value in point)


def _neighbor_keys(key):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (key[0] + dx, key[1] + dy, key[2] + dz)


def directional_vertex_coverage(
    reference_points: np.ndarray,
    observed_points: np.ndarray,
    reference_weights: np.ndarray,
    observed_weights: np.ndarray,
    *,
    maximum_position_delta: float,
    maximum_weight_delta: float,
    label: str,
) -> dict:
    """Prove every observed serialized vertex derives from a reference vertex.

    glTF export may split a Blender vertex at an attribute seam, so output and
    input vertex indices and counts are not stable identities.  A fixed-radius
    spatial grid plus the complete bone-weight vector provides a stronger
    identity that admits only genuine serialization duplicates.
    """

    if (
        reference_points.ndim != 2
        or observed_points.ndim != 2
        or reference_points.shape[1:] != (3,)
        or observed_points.shape[1:] != (3,)
        or reference_weights.shape[0] != len(reference_points)
        or observed_weights.shape[0] != len(observed_points)
        or reference_weights.shape[1] != observed_weights.shape[1]
    ):
        raise RuntimeError(f"{label} vertex/weight arrays are incompatible")
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(reference_points):
        grid.setdefault(_grid_key(point, maximum_position_delta), []).append(
            index
        )
    maximum_observed_position_delta = 0.0
    maximum_observed_weight_delta = 0.0
    selected_reference_indices = []
    for observed_index, point in enumerate(observed_points):
        candidates = []
        for key in _neighbor_keys(
            _grid_key(point, maximum_position_delta)
        ):
            for reference_index in grid.get(key, ()):
                position_delta = float(
                    np.linalg.norm(point - reference_points[reference_index])
                )
                if position_delta > maximum_position_delta:
                    continue
                weight_delta = float(
                    np.max(
                        np.abs(
                            observed_weights[observed_index]
                            - reference_weights[reference_index]
                        ),
                        initial=0.0,
                    )
                )
                if weight_delta <= maximum_weight_delta:
                    candidates.append(
                        (weight_delta, position_delta, reference_index)
                    )
        if not candidates:
            raise RuntimeError(
                f"{label} vertex {observed_index} has no geometry-and-weight "
                "match in the authenticated artifact"
            )
        weight_delta, position_delta, reference_index = min(candidates)
        maximum_observed_position_delta = max(
            maximum_observed_position_delta, position_delta
        )
        maximum_observed_weight_delta = max(
            maximum_observed_weight_delta, weight_delta
        )
        selected_reference_indices.append(reference_index)
    return {
        "maximum_position_delta": maximum_observed_position_delta,
        "maximum_weight_delta": maximum_observed_weight_delta,
        "selected_reference_indices": selected_reference_indices,
    }


def main():
    args = parse_argv()
    pre_path = require_input(args.pre_level_input, "pre-level input GLB")
    output_path = require_input(args.leveled_output, "leveled output GLB")
    manifest_path = require_input(args.support_manifest, "support manifest")
    rig_audit_path = require_input(args.rig_audit, "rig audit")
    readback_path = require_new_output(args.readback_output)
    manifest = load_json(manifest_path, "support manifest")
    rig_audit = load_json(rig_audit_path, "rig audit")
    support = manifest.get("support_plane")
    if not isinstance(support, dict):
        raise RuntimeError("support manifest has no measurements")
    transform = validate_rigid_leveling_transform(
        support.get("dual_authority"), support
    )

    pre = snapshot(pre_path, args.front_axis)
    post = snapshot(output_path, args.front_axis)
    post_diagonal_ratio_delta = abs(
        post["mesh_diagonal"] - pre["mesh_diagonal"]
    ) / pre["mesh_diagonal"]
    if (
        post_diagonal_ratio_delta
        > MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA
    ):
        raise RuntimeError(
            "post-level bounding-box diagonal changed beyond the rigid "
            f"rotation allowance: ratio={post_diagonal_ratio_delta}"
        )
    scale = max(1.0, pre["mesh_diagonal"])
    maximum_vertex_delta = MAXIMUM_RIGID_VERTEX_DELTA_RATIO * scale
    maximum_bone_delta = MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO * scale
    rotation = transform["rotation_from_primary_normal_to_positive_z"]
    translation = transform["vertical_translation"]

    if pre["semantic_rig"] != post["semantic_rig"]:
        raise RuntimeError("semantic foot/limb ownership changed after export")
    if support.get("foot_leaves") != pre["semantic_rig"]["foot_leaves"]:
        raise RuntimeError(
            "support manifest foot leaves do not match independent inference"
        )
    if pre["scene"] != manifest.get("scene_before"):
        raise RuntimeError(
            "support manifest pre-level scene does not match independent import"
        )
    if post["scene"] != manifest.get("scene_after"):
        raise RuntimeError(
            "support manifest post-level scene does not match independent import"
        )
    reviewed_bones = rig_audit.get("armature", {}).get("bone_count")
    if (
        isinstance(reviewed_bones, bool)
        or not isinstance(reviewed_bones, int)
        or reviewed_bones != pre["scene"]["bone_count"]
        or reviewed_bones != post["scene"]["bone_count"]
    ):
        raise RuntimeError("independent readback bone count changed from rig audit")
    if pre["polygon_count"] != post["polygon_count"]:
        raise RuntimeError(
            "mesh topology changed during support-plane export: "
            f"vertices={len(pre['vertices'])}/{len(post['vertices'])} "
            f"polygons={pre['polygon_count']}/{post['polygon_count']} "
            f"topology={pre['topology_sha256']}/{post['topology_sha256']}"
        )
    expected_vertices = transform_points(
        pre["vertices"], rotation, translation
    )
    if pre["bone_names"] != post["bone_names"]:
        raise RuntimeError("skin-weight bone groups changed during export")
    expansion_ratio = (
        len(post["vertices"]) - len(pre["vertices"])
    ) / len(pre["vertices"])
    if (
        expansion_ratio < 0.0
        or expansion_ratio > MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO
    ):
        raise RuntimeError(
            "support-plane export vertex count is not a bounded glTF seam split: "
            f"before={len(pre['vertices'])} after={len(post['vertices'])} "
            f"ratio={expansion_ratio}"
        )
    forward_coverage = directional_vertex_coverage(
        expected_vertices,
        post["vertices"],
        pre["weights"],
        post["weights"],
        maximum_position_delta=maximum_vertex_delta,
        maximum_weight_delta=MAXIMUM_SKIN_WEIGHT_DELTA,
        label="leveled output",
    )
    reverse_coverage = directional_vertex_coverage(
        post["vertices"],
        expected_vertices,
        post["weights"],
        pre["weights"],
        maximum_position_delta=maximum_vertex_delta,
        maximum_weight_delta=MAXIMUM_SKIN_WEIGHT_DELTA,
        label="authenticated pre-level input",
    )
    vertex_delta = max(
        forward_coverage["maximum_position_delta"],
        reverse_coverage["maximum_position_delta"],
    )

    pre_records = {record["name"]: record for record in pre["records"]}
    post_records = {record["name"]: record for record in post["records"]}
    if set(pre_records) != set(post_records):
        raise RuntimeError("skeleton bone names changed during support-plane export")
    maximum_endpoint_delta = 0.0
    for name in sorted(pre_records):
        before = pre_records[name]
        after = post_records[name]
        if (
            before["parent"] != after["parent"]
            or before["children"] != after["children"]
        ):
            raise RuntimeError(
                f"skeleton hierarchy changed during support-plane export: {name}"
            )
        for endpoint in ("head_world", "tail_world"):
            expected = transform_points(
                np.asarray([before[endpoint]], dtype=np.float64),
                rotation,
                translation,
            )
            maximum_endpoint_delta = max(
                maximum_endpoint_delta,
                maximum_point_delta(
                    [after[endpoint]], expected, f"{name} {endpoint}"
                ),
            )
    if maximum_endpoint_delta > maximum_bone_delta:
        raise RuntimeError(
            "leveled output skeleton does not contain the declared rigid transform: "
            f"delta={maximum_endpoint_delta} maximum={maximum_bone_delta}"
        )

    weight_delta = max(
        forward_coverage["maximum_weight_delta"],
        reverse_coverage["maximum_weight_delta"],
    )
    changed_weights = 0
    if pre["materials"] != post["materials"] or pre["uv_layers"] != post["uv_layers"]:
        raise RuntimeError("mesh material or UV authority changed during export")

    pre_primary_delta = maximum_point_delta(
        pre["dual_authority"]["primary"]["foot_points"],
        support["dual_authority"]["primary"]["foot_points"],
        "pre-level primary foot readback",
    )
    pre_cross_delta = maximum_point_delta(
        pre["dual_authority"]["crosscheck"]["foot_points"],
        support["dual_authority"]["crosscheck"]["foot_points"],
        "pre-level crosscheck foot readback",
    )
    post_primary_delta = maximum_point_delta(
        post["dual_authority"]["primary"]["foot_points"],
        support["foot_points_after"],
        "post-level primary foot readback",
    )
    post_cross_delta = maximum_point_delta(
        post["dual_authority"]["crosscheck"]["foot_points"],
        support["crosscheck_foot_points_after"],
        "post-level crosscheck foot readback",
    )
    foot_tolerance = transform["transform_absolute_tolerance"]
    post_semantic_reacquisition_tolerance = transform["capture_radius"]
    if (
        max(pre_primary_delta, pre_cross_delta) > foot_tolerance
        or max(post_primary_delta, post_cross_delta)
        > post_semantic_reacquisition_tolerance
    ):
        raise RuntimeError(
            "support-plane foot evidence does not match independent GLB import: "
            f"pre_primary={pre_primary_delta} pre_cross={pre_cross_delta} "
            f"post_primary={post_primary_delta} post_cross={post_cross_delta} "
            f"pre_maximum={foot_tolerance} "
            "post_semantic_reacquisition_maximum="
            f"{post_semantic_reacquisition_tolerance}"
        )
    actual_minimum_z = min(
        point[2]
        for point in post["dual_authority"]["primary"]["foot_points"]
    )
    if not math.isclose(
        actual_minimum_z,
        float(support["minimum_foot_z_after"]),
        rel_tol=1.0e-7,
        abs_tol=transform["band_thickness"],
    ):
        raise RuntimeError(
            "support-plane minimum foot Z does not match independent GLB import: "
            f"actual={actual_minimum_z} "
            f"declared={support['minimum_foot_z_after']} "
            f"maximum={transform['band_thickness']}"
        )
    post_tilt = float(
        post["dual_authority"]["primary"]["plane"]["tilt_deg"]
    )
    if post_tilt > MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG:
        raise RuntimeError(
            "independently imported primary support plane is not level: "
            f"tilt={post_tilt}"
        )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_independent_glb_reimport",
        "formal_dataset_registration_authorized": False,
        "pre_level_input": file_record(pre_path),
        "leveled_output": file_record(output_path),
        "support_manifest": path_hash_record(manifest_path),
        "rig_audit": path_hash_record(rig_audit_path),
        "front_axis": args.front_axis,
        "thresholds": {
            "maximum_rigid_vertex_delta_ratio_of_mesh_diagonal": (
                MAXIMUM_RIGID_VERTEX_DELTA_RATIO
            ),
            "maximum_rigid_vertex_delta": maximum_vertex_delta,
            "maximum_rigid_bone_endpoint_delta_ratio_of_mesh_diagonal": (
                MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
            ),
            "maximum_rigid_bone_endpoint_delta": maximum_bone_delta,
            "maximum_skin_weight_delta": MAXIMUM_SKIN_WEIGHT_DELTA,
            "maximum_primary_post_level_tilt_deg": (
                MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG
            ),
            "maximum_serialization_vertex_expansion_ratio": (
                MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO
            ),
            "maximum_foot_readback_delta": foot_tolerance,
            "maximum_post_level_semantic_reacquisition_delta": (
                post_semantic_reacquisition_tolerance
            ),
            "maximum_post_level_floor_reacquisition_delta": transform[
                "band_thickness"
            ],
            "maximum_post_level_bbox_diagonal_ratio_delta": (
                MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA
            ),
        },
        "pre_level_scene": pre["scene"],
        "post_level_scene": post["scene"],
        "pre_level_semantic_rig": pre["semantic_rig"],
        "post_level_semantic_rig": post["semantic_rig"],
        "pre_level_dual_authority": pre["dual_authority"],
        "post_level_dual_authority": post["dual_authority"],
        "object_transform_readback": {
            "pre_mesh_world_matrix": pre["mesh_world_matrix"],
            "post_mesh_world_matrix": post["mesh_world_matrix"],
            "pre_armature_world_matrix": pre["armature_world_matrix"],
            "post_armature_world_matrix": post["armature_world_matrix"],
            "pre_root_objects": pre["root_objects"],
            "post_root_objects": post["root_objects"],
        },
        "mesh_and_weight_binding": {
            "vertex_count": len(pre["vertices"]),
            "serialized_vertex_count": len(post["vertices"]),
            "serialization_vertex_expansion_ratio": expansion_ratio,
            "polygon_count": pre["polygon_count"],
            "topology_sha256_before": pre["topology_sha256"],
            "topology_sha256_after": post["topology_sha256"],
            "bone_group_names": pre["bone_names"],
            "maximum_skin_weight_delta": weight_delta,
            "weights_changed_above_tolerance": changed_weights,
            "materials": pre["materials"],
            "uv_layers": pre["uv_layers"],
        },
        "comparison": {
            "maximum_world_vertex_delta_from_declared_transform": vertex_delta,
            "maximum_bone_endpoint_delta_from_declared_transform": (
                maximum_endpoint_delta
            ),
            "pre_level_primary_foot_delta": pre_primary_delta,
            "pre_level_crosscheck_foot_delta": pre_cross_delta,
            "post_level_primary_foot_delta": post_primary_delta,
            "post_level_crosscheck_foot_delta": post_cross_delta,
            "actual_minimum_primary_foot_z": actual_minimum_z,
            "primary_post_level_tilt_deg": post_tilt,
            "post_level_bbox_diagonal_ratio_delta": (
                post_diagonal_ratio_delta
            ),
            "passed": True,
        },
    }
    with readback_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_SUPPORT_PLANE_READBACK_OK "
        f"vertex_delta={vertex_delta:.9g} weight_delta={weight_delta:.9g} "
        f"output={readback_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
