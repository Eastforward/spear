#!/usr/bin/env python3
"""Measure quadruped in/out gait at hips, paws, and terminal paw yaw."""

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
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.blender_audit_generated_quadruped_foot_contact import (  # noqa: E402
    evaluated_vertices,
    hidden_objects,
    linked_armatures,
    resolve_action,
    target_records,
    weighted_skin_bone_names,
)
from tools.generated_quadruped_semantics import infer_quadruped_semantics  # noqa: E402


SCHEMA = "avengine_quadruped_lateral_gait_audit_v1"
AXES = {
    "positive-x": Vector((1.0, 0.0, 0.0)),
    "negative-x": Vector((-1.0, 0.0, 0.0)),
    "positive-y": Vector((0.0, 1.0, 0.0)),
    "negative-y": Vector((0.0, -1.0, 0.0)),
}


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-axis", choices=tuple(AXES), required=True)
    parser.add_argument("--action", default="Walking")
    parser.add_argument("--samples", type=int, default=41)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_head(armature, name):
    return armature.matrix_world @ armature.pose.bones[name].head


def world_tail(armature, name):
    return armature.matrix_world @ armature.pose.bones[name].tail


def lateral_value(point, lateral):
    return float(point.dot(lateral))


def range_value(values):
    return float(max(values) - min(values))


def main():
    args = parse_argv()
    source = args.input.absolute()
    output = args.output.absolute()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 8 <= args.samples <= 241:
        raise SystemExit("--samples must be in [8, 241]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    hidden = hidden_objects()
    meshes = [
        item
        for item in bpy.data.objects
        if item.type == "MESH" and item not in hidden and linked_armatures(item)
    ]
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise SystemExit("input must contain one real skinned mesh and one armature")
    body, armature = meshes[0], armatures[0]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    armature.animation_data_clear()
    armature.data.pose_position = "REST"
    scene.frame_set(0)
    bpy.context.view_layer.update()
    rest_vertices = evaluated_vertices(body, depsgraph)
    bbox_min = rest_vertices.min(axis=0)
    bbox_extent = np.ptp(rest_vertices, axis=0)
    diagonal = float(np.linalg.norm(bbox_extent))
    included = weighted_skin_bone_names(body, armature)
    semantics = infer_quadruped_semantics(
        target_records(armature, included),
        bbox_min=bbox_min,
        bbox_extent=bbox_extent,
        front_axis=args.front_axis,
    )
    chains = {
        "front_side_negative": semantics.front_side_negative,
        "front_side_positive": semantics.front_side_positive,
        "hind_side_negative": semantics.hind_side_negative,
        "hind_side_positive": semantics.hind_side_positive,
    }
    forward = AXES[args.front_axis]
    up = Vector((0.0, 0.0, 1.0))
    lateral = up.cross(forward).normalized()
    rest = {}
    for label, chain in chains.items():
        hip = world_head(armature, chain[0])
        paw = world_tail(armature, chain[-1])
        rest[label] = {
            "chain": list(chain),
            "hip_lateral": lateral_value(hip, lateral),
            "paw_lateral": lateral_value(paw, lateral),
            "paw_relative_to_hip_lateral": lateral_value(paw - hip, lateral),
        }

    action = resolve_action(args.action, list(bpy.data.actions))
    armature.data.pose_position = "POSE"
    armature.animation_data_create()
    armature.animation_data.action = action
    start, end = map(float, action.frame_range)
    frames = []
    for value in np.linspace(start, end, args.samples):
        scene.frame_set(int(round(float(value))))
        bpy.context.view_layer.update()
        limbs = {}
        for label, chain in chains.items():
            hip = world_head(armature, chain[0])
            paw_head = world_head(armature, chain[-1])
            paw_tail = world_tail(armature, chain[-1])
            paw_vector = paw_tail - paw_head
            horizontal = paw_vector - up * paw_vector.dot(up)
            paw_yaw = None
            if horizontal.length > 1.0e-8:
                horizontal.normalize()
                paw_yaw = math.degrees(
                    math.atan2(horizontal.dot(lateral), horizontal.dot(forward))
                )
            limbs[label] = {
                "hip_lateral": lateral_value(hip, lateral),
                "paw_lateral": lateral_value(paw_tail, lateral),
                "paw_relative_to_hip_lateral": lateral_value(
                    paw_tail - hip, lateral
                ),
                "paw_yaw_degrees": paw_yaw,
            }
        frames.append({"source_frame": float(value), "limbs": limbs})

    summary = {}
    for label in chains:
        hip = [item["limbs"][label]["hip_lateral"] for item in frames]
        paw = [item["limbs"][label]["paw_lateral"] for item in frames]
        relative = [
            item["limbs"][label]["paw_relative_to_hip_lateral"]
            for item in frames
        ]
        yaw = [
            item["limbs"][label]["paw_yaw_degrees"]
            for item in frames
            if item["limbs"][label]["paw_yaw_degrees"] is not None
        ]
        summary[label] = {
            "hip_lateral_excursion": range_value(hip),
            "paw_lateral_excursion": range_value(paw),
            "paw_relative_to_hip_lateral_excursion": range_value(relative),
            "paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal": (
                range_value(relative) / diagonal
            ),
            "paw_yaw_range_degrees": [float(min(yaw)), float(max(yaw))],
            "paw_yaw_excursion_degrees": range_value(yaw),
        }

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "coordinate_contract": {
            "front_axis": args.front_axis,
            "up_axis": "positive-z",
            "lateral_axis_vector": [float(value) for value in lateral],
        },
        "action": action.name,
        "frame_range": [start, end],
        "mesh_diagonal": diagonal,
        "rest": rest,
        "sampled_frames": frames,
        "summary": summary,
        "formal_dataset_registration_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"QUADRUPED_LATERAL_GAIT_AUDIT_OK output={output}", flush=True)


if __name__ == "__main__":
    main()
