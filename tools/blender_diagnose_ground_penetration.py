#!/usr/bin/env python3
"""Trace animated ground penetration to exact vertices and skin influences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np


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


SCHEMA = "avengine_ground_penetration_diagnostic_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", default="Walking")
    parser.add_argument("--front-axis", required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertex_influences(mesh, vertex_index, labels):
    groups = {group.index: group.name for group in mesh.vertex_groups}
    values = []
    for membership in mesh.data.vertices[int(vertex_index)].groups:
        name = groups.get(membership.group)
        if name not in labels or membership.weight <= 1.0e-6:
            continue
        values.append(
            {
                "bone": name,
                "semantic_chain": labels[name],
                "weight": float(membership.weight),
            }
        )
    values.sort(key=lambda item: (-item["weight"], item["bone"]))
    return values


def main():
    args = parse_argv()
    source = args.input.absolute()
    output = args.output.absolute()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 4 <= args.samples <= 120 or not 1 <= args.top_k <= 512:
        raise SystemExit("diagnostic sampling parameters are out of range")
    output.parent.mkdir(parents=True, exist_ok=True)

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
    mesh = meshes[0]
    armature = armatures[0]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    available = list(bpy.data.actions)
    action = resolve_action(args.action, available)

    armature.animation_data_clear()
    armature.data.pose_position = "REST"
    scene.frame_set(1)
    bpy.context.view_layer.update()
    rest = evaluated_vertices(mesh, depsgraph)
    rest_floor = float(rest[:, 2].min())
    semantic_bones = weighted_skin_bone_names(mesh, armature)
    semantics = infer_quadruped_semantics(
        target_records(armature, semantic_bones),
        bbox_min=rest.min(axis=0),
        bbox_extent=np.ptp(rest, axis=0),
        front_axis=args.front_axis,
    )
    labels = {}
    for label, chain in semantics.chains().items():
        for name in chain:
            labels[name] = label

    armature.data.pose_position = "POSE"
    armature.animation_data_create()
    armature.animation_data.action = action
    start, end = [float(value) for value in action.frame_range]
    candidates = []
    for value in np.linspace(start, end, args.samples):
        base = int(value)
        scene.frame_set(base, subframe=float(value - base))
        bpy.context.view_layer.update()
        posed = evaluated_vertices(mesh, depsgraph)
        count = min(args.top_k, len(posed))
        indices = np.argpartition(posed[:, 2], count - 1)[:count]
        for index in indices:
            candidates.append(
                {
                    "source_frame": float(value),
                    "evaluated_frame": int(base),
                    "vertex_index": int(index),
                    "rest_world": [float(item) for item in rest[index]],
                    "posed_world": [float(item) for item in posed[index]],
                    "ground_delta": float(posed[index, 2] - rest_floor),
                    "vertex_vertical_displacement": float(
                        posed[index, 2] - rest[index, 2]
                    ),
                }
            )
    candidates.sort(key=lambda item: item["ground_delta"])
    selected = candidates[: args.top_k]
    for item in selected:
        item["influences"] = vertex_influences(
            mesh,
            item["vertex_index"],
            labels,
        )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "action": {
            "requested": args.action,
            "resolved": action.name,
            "frame_range": [start, end],
            "samples": args.samples,
        },
        "rest_floor_world_z": rest_floor,
        "minimum_ground_delta": selected[0]["ground_delta"],
        "minimum_vertex": selected[0],
        "lowest_samples": selected,
        "formal_dataset_registration_authorized": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GROUND_PENETRATION_DIAGNOSTIC_OK "
        f"minimum={payload['minimum_ground_delta']:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
