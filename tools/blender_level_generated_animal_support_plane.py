"""Level a generated quadruped from its four semantic foot endpoints.

Image-to-3D output can be anatomically usable while the complete animal is
exported with a non-zero pitch or roll.  After heading normalization and
target-native rigging, the four foot chains provide stable support evidence.
This stage fits one least-squares foot plane, rigidly rotates every scene root
so its normal becomes world +Z, and translates the lowest reviewed foot to
z=0.  Mesh topology, materials, hierarchy, and skin weights are untouched.
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
from mathutils import Matrix, Vector
import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_quadruped_semantics import infer_quadruped_semantics


SCHEMA = "avengine_generated_animal_support_plane_leveling_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--front-axis",
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        required=True,
    )
    parser.add_argument("--review-evidence", type=Path, required=True)
    parser.add_argument(
        "--plane-source",
        choices=("bone-endpoints", "mesh-foot-bottoms"),
        default="bone-endpoints",
        help=(
            "bone-endpoints levels on the semantic foot-chain bone tips "
            "(v1 behavior).  mesh-foot-bottoms levels on the visible mesh "
            "ground-contact band around each foot instead: generated rigs "
            "can extend bone tails unevenly below the mesh, leaving the "
            "asset floating and pitched even though the bone tips are "
            "perfectly level.  Both modes apply one rigid rotation plus one "
            "vertical translation and never move feet individually."
        ),
    )
    parser.add_argument("--maximum-tilt-deg", type=float, default=30.0)
    parser.add_argument(
        "--maximum-foot-plane-residual-ratio",
        type=float,
        default=0.02,
        help=(
            "Reject when any semantic foot differs from the fitted support "
            "plane by more than this fraction of the target mesh diagonal."
        ),
    )
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


def require_new_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def mesh_bbox(mesh):
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = np.asarray(
        [min(point[axis] for point in points) for axis in range(3)],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [max(point[axis] for point in points) for axis in range(3)],
        dtype=np.float64,
    )
    return minimum, maximum - minimum


def semantic_records(armature):
    records = []
    for bone in armature.data.bones:
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        records.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "children": [child.name for child in bone.children],
                "head_world": [float(value) for value in head],
                "tail_world": [float(value) for value in tail],
            }
        )
    return records


def lower_endpoint(record):
    head = np.asarray(record["head_world"], dtype=np.float64)
    tail = np.asarray(record["tail_world"], dtype=np.float64)
    return head if head[2] <= tail[2] else tail


def mesh_foot_bottom_point(world_vertices, anchor, mesh_diagonal):
    """Ground-contact point of one foot: among mesh vertices within a
    horizontal capture radius of the foot-bone anchor, take the lowest-z
    vertex band and return its centroid at the true minimum height.  Falls
    back to the bone anchor when the capture region is empty."""
    horizontal = np.linalg.norm(world_vertices[:, :2] - anchor[:2], axis=1)
    capture = world_vertices[horizontal < mesh_diagonal*0.05]
    if len(capture) < 10:
        return np.asarray(anchor, dtype=np.float64), 0
    z_floor = float(capture[:, 2].min())
    band = capture[capture[:, 2] <= z_floor + max(0.004, mesh_diagonal*0.003)]
    return (
        np.asarray(
            [float(band[:, 0].mean()), float(band[:, 1].mean()), z_floor],
            dtype=np.float64,
        ),
        int(len(band)),
    )


def scene_summary():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
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
    }, skinned, armatures


def main():
    args = parse_argv()
    source = require_input(args.input, "heading-normalized rigged GLB")
    evidence = require_input(args.review_evidence, "heading/rig review evidence")
    output = require_new_output(args.output, "leveled GLB")
    manifest = require_new_output(args.manifest, "leveling manifest")
    if not 0.0 < args.maximum_tilt_deg <= 45.0:
        raise SystemExit("--maximum-tilt-deg must be in (0, 45]")
    if not 0.0 < args.maximum_foot_plane_residual_ratio <= 0.1:
        raise SystemExit("--maximum-foot-plane-residual-ratio must be in (0, 0.1]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    before, skinned, armatures = scene_summary()
    if len(skinned) != 1 or len(armatures) != 1:
        raise RuntimeError(f"expected one skinned mesh and one armature: {before}")
    if before["action_count"] != 0:
        raise RuntimeError("support-plane leveling must run before animation")
    mesh = skinned[0]
    armature = armatures[0]
    minimum, extent = mesh_bbox(mesh)
    records = semantic_records(armature)
    semantics = infer_quadruped_semantics(
        records,
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=args.front_axis,
    )
    by_name = {record["name"]: record for record in records}
    bone_anchor_points = np.asarray(
        [lower_endpoint(by_name[name]) for name in semantics.foot_leaves],
        dtype=np.float64,
    )
    foot_point_band_sizes = None
    if args.plane_source == "mesh-foot-bottoms":
        world_matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
        local = np.empty((len(mesh.data.vertices), 3), dtype=np.float64)
        mesh.data.vertices.foreach_get("co", local.ravel())
        world_vertices = local @ world_matrix[:3, :3].T + world_matrix[:3, 3]
        mesh_diagonal_early = float(np.linalg.norm(extent))
        resolved = [
            mesh_foot_bottom_point(world_vertices, anchor, mesh_diagonal_early)
            for anchor in bone_anchor_points
        ]
        foot_points = np.asarray([point for point, _band in resolved])
        foot_point_band_sizes = [band for _point, band in resolved]
    else:
        foot_points = bone_anchor_points
    design = np.column_stack((foot_points[:, 0], foot_points[:, 1], np.ones(4)))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design,
        foot_points[:, 2],
        rcond=None,
    )
    predicted = design @ coefficients
    residuals = foot_points[:, 2] - predicted
    mesh_diagonal = float(np.linalg.norm(extent))
    if mesh_diagonal <= 0.0:
        raise RuntimeError("target mesh has zero diagonal")
    maximum_residual = float(np.abs(residuals).max())
    maximum_residual_ratio = maximum_residual / mesh_diagonal
    if maximum_residual_ratio > args.maximum_foot_plane_residual_ratio:
        raise RuntimeError(
            "semantic feet do not define one support plane: residual ratio "
            f"{maximum_residual_ratio:.6f} exceeds reviewed maximum "
            f"{args.maximum_foot_plane_residual_ratio:.6f}"
        )
    normal = Vector(
        (-float(coefficients[0]), -float(coefficients[1]), 1.0)
    ).normalized()
    up = Vector((0.0, 0.0, 1.0))
    tilt_deg = math.degrees(normal.angle(up))
    if tilt_deg > args.maximum_tilt_deg:
        raise RuntimeError(
            f"support plane tilt {tilt_deg:.6f} exceeds reviewed maximum "
            f"{args.maximum_tilt_deg:.6f}"
        )
    rotation = normal.rotation_difference(up).to_matrix().to_4x4()
    rotated_feet = np.asarray(
        [tuple(rotation @ Vector(point)) for point in foot_points],
        dtype=np.float64,
    )
    vertical_translation = -float(rotated_feet[:, 2].min())
    transform = Matrix.Translation((0.0, 0.0, vertical_translation)) @ rotation
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not roots:
        raise RuntimeError("imported scene has no root objects")
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    post_feet = np.asarray(
        [tuple(transform @ Vector(point)) for point in foot_points],
        dtype=np.float64,
    )

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
    after, _skinned_after, _armatures_after = scene_summary()
    for key in (
        "mesh_count",
        "skinned_mesh_count",
        "armature_count",
        "bone_count",
        "material_count",
        "image_count",
        "action_count",
    ):
        if after[key] != before[key]:
            raise RuntimeError(f"rigid leveling changed {key}: {before} -> {after}")

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "technical_spike_only_pending_retarget_and_visual_qa",
        "formal_dataset_registration_authorized": False,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "review_evidence": {
            "path": str(evidence),
            "sha256": sha256_file(evidence),
        },
        "support_plane": {
            "front_axis": args.front_axis,
            "foot_leaves": list(semantics.foot_leaves),
            "plane_source": args.plane_source,
            "mesh_foot_contact_band_sizes": foot_point_band_sizes,
            "foot_points_before": foot_points.tolist(),
            "z_equals_ax_plus_by_plus_c": coefficients.tolist(),
            "residual_z": residuals.tolist(),
            "maximum_residual": maximum_residual,
            "maximum_residual_ratio_of_mesh_diagonal": maximum_residual_ratio,
            "maximum_reviewed_residual_ratio_of_mesh_diagonal": (
                args.maximum_foot_plane_residual_ratio
            ),
            "normal_before": list(normal),
            "tilt_deg": tilt_deg,
            "maximum_tilt_deg": args.maximum_tilt_deg,
            "applied_vertical_translation": vertical_translation,
            "foot_points_after": post_feet.tolist(),
            "minimum_foot_z_after": float(post_feet[:, 2].min()),
            "policy": "four_semantic_feet_least_squares_plane_rigid_leveling",
        },
        "preservation_contract": {
            "mesh_topology_changed": False,
            "material_changed": False,
            "skeleton_hierarchy_changed": False,
            "skin_weights_changed": False,
            "animation_present_or_changed": False,
        },
        "scene_before": before,
        "scene_after": after,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_SUPPORT_PLANE_LEVELING_OK "
        f"tilt_deg={tilt_deg:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
