#!/usr/bin/env python3
"""Build a skeleton-only motion-basis review before target animation exists.

The target native mesh and fitted rest skeleton are displayed as immutable
reference geometry.  Quaternius Walking is sampled in memory and mapped onto
the target *pose skeleton only* for every reviewer-selectable cardinal yaw and
lateral-chain mapping.  This tool never exports a GLB and never writes target
animation keyframes; its JSON is consumed by the interactive human gate.
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
from mathutils import Matrix, Quaternion, Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_retarget_quaternius_to_generated_quadruped as retarget  # noqa: E402


SCHEMA = "avengine_generated_animal_motion_basis_preview_v1"
CARDINAL_MOTION_BASIS_YAWS = retarget.CARDINAL_MOTION_BASIS_YAWS


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--source-rig-glb", type=Path, required=True)
    parser.add_argument(
        "--target-front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-mesh-points", type=int, default=6000)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_output(path: Path) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rounded_vector(value, digits=6):
    return [round(float(component), digits) for component in value]


def mesh_sample_points(mesh, maximum):
    vertices = mesh.data.vertices
    if maximum <= 0:
        raise RuntimeError("--max-mesh-points must be positive")
    stride = max(1, math.ceil(len(vertices) / maximum))
    return [
        rounded_vector(mesh.matrix_world @ vertices[index].co)
        for index in range(0, len(vertices), stride)
    ]


def bone_segments(armature, *, pose):
    bones = armature.pose.bones if pose else armature.data.bones
    segments = []
    for bone in bones:
        if pose:
            head = armature.matrix_world @ bone.head
            tail = armature.matrix_world @ bone.tail
        else:
            head = armature.matrix_world @ bone.head_local
            tail = armature.matrix_world @ bone.tail_local
        segments.append(rounded_vector(head) + rounded_vector(tail))
    return segments


def apply_preview_pose(
    target,
    cached_frame,
    plan,
    *,
    motion_basis_yaw_deg,
):
    by_target = {entry["target"]: entry for entry in plan}
    order = retarget.target_parent_first(target)
    rest_locals = {
        name: retarget.parent_local_rest(target.data.bones[name])
        for name in target.data.bones.keys()
    }
    target_object_rotation = target.matrix_world.to_quaternion().normalized()
    basis_rotation = Quaternion(
        Vector((0.0, 0.0, 1.0)), math.radians(float(motion_basis_yaw_deg))
    ).normalized()
    for pose_bone in target.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for target_name in order:
        entry = by_target[target_name]
        pose_bone = target.pose.bones[target_name]
        rest_local = rest_locals[target_name]
        if pose_bone.parent is None:
            translation = rest_local.translation.copy()
        else:
            translation = pose_bone.parent.matrix @ rest_local.translation
        source_first = cached_frame["rotations"][entry["source_first"]]
        source_second = cached_frame["rotations"][entry["source_second"]]
        source_pose_world = source_first.slerp(source_second, entry["blend"])
        source_delta_world = (
            source_pose_world @ entry["source_rest_world"].inverted()
        ).normalized()
        source_delta_world = (
            basis_rotation
            @ source_delta_world
            @ basis_rotation.inverted()
        ).normalized()
        desired_world = (
            source_delta_world @ entry["target_rest_world"]
        ).normalized()
        desired_armature = (
            target_object_rotation.inverted() @ desired_world
        ).normalized()
        pose_bone.matrix = Matrix.LocRotScale(
            translation, desired_armature, Vector((1.0, 1.0, 1.0))
        )
        bpy.context.view_layer.update()


def candidate_payload(
    target,
    source,
    source_action,
    semantics,
    chains,
    *,
    motion_basis_yaw_deg,
    side_chain_mode,
):
    plan = retarget.full_sampling_plan(
        target,
        source,
        semantics,
        chains,
        side_chain_mode=side_chain_mode,
    )
    source_bones = sorted(
        {entry["source_first"] for entry in plan}
        | {entry["source_second"] for entry in plan}
    )
    cached = retarget.cache_source_action(source, source_action, source_bones)
    frames = []
    for cached_frame in cached:
        apply_preview_pose(
            target,
            cached_frame,
            plan,
            motion_basis_yaw_deg=motion_basis_yaw_deg,
        )
        frames.append(
            {
                "frame": int(cached_frame["frame"]),
                "source_frame": round(float(cached_frame["source_frame"]), 6),
                "segments": bone_segments(target, pose=True),
            }
        )
    angle = math.radians(float(motion_basis_yaw_deg))
    candidate_id = (
        f"yaw_{motion_basis_yaw_deg:+d}_side_{side_chain_mode}"
        .replace("+", "p")
        .replace("-", "m")
    )
    return {
        "candidate_id": candidate_id,
        "motion_basis_yaw_deg": int(motion_basis_yaw_deg),
        "side_chain_mode": side_chain_mode,
        "rotation_transfer_mode": "world-left-delta-v2",
        "source_motion_forward": [
            round(math.cos(angle), 6),
            round(math.sin(angle), 6),
            0.0,
        ],
        "target_animation_generated": False,
        "frames": frames,
    }


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main():
    args = parse_argv()
    target_path = require_file(args.target_glb, "target GLB")
    source_path = require_file(args.source_rig_glb, "source rig GLB")
    output = require_output(args.output_json)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    target, mesh, _target_imported = retarget.import_target(target_path)
    retarget.detach_armature_parent(target)
    target.data.pose_position = "REST"
    bpy.context.view_layer.update()
    minimum, maximum, extent = retarget.target_bbox(mesh)
    semantics, chains = retarget.target_chains(
        target, mesh, args.target_front_axis
    )
    target_rest_segments = bone_segments(target, pose=False)
    points = mesh_sample_points(mesh, args.max_mesh_points)
    bone_order = [bone.name for bone in target.data.bones]
    source, _source_imported, source_actions = retarget.import_source(source_path)

    candidates = []
    for side_chain_mode in ("matched", "swapped"):
        for yaw in CARDINAL_MOTION_BASIS_YAWS:
            candidates.append(
                candidate_payload(
                    target,
                    source,
                    source_actions["Walking"],
                    semantics,
                    chains,
                    motion_basis_yaw_deg=yaw,
                    side_chain_mode=side_chain_mode,
                )
            )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "pre_animation_review_only",
        "asset_id": args.asset_id,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "target_animation_generated": False,
        "target": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "size_bytes": target_path.stat().st_size,
            "reviewed_front_axis": args.target_front_axis,
            "canonical_forward": [1.0, 0.0, 0.0],
            "bbox_min": rounded_vector(minimum),
            "bbox_max": rounded_vector(maximum),
            "bbox_extent": rounded_vector(extent),
            "mesh_points": points,
            "bone_order": bone_order,
            "target_rest_segments": target_rest_segments,
            "semantic_chains": {
                name: list(value) for name, value in chains.items()
            },
            "foot_leaves": list(semantics.foot_leaves),
        },
        "source_motion": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
            "action": source_actions["Walking"].name,
            "geometry_used": False,
            "weights_used": False,
        },
        "review_contract": {
            "allowed_motion_basis_yaws_deg": list(CARDINAL_MOTION_BASIS_YAWS),
            "allowed_side_chain_modes": ["matched", "swapped"],
            "required_before_target_animation": True,
            "fine_yaw_allowed": False,
        },
        "candidates": candidates,
    }
    payload["preview_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_MOTION_BASIS_PREVIEW_OK "
        f"asset={args.asset_id} candidates={len(candidates)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
