#!/usr/bin/env python3
"""Audit generated quadruped Walk/Idle grounding without bone-name assumptions.

The reviewed cardinal front axis and rest hierarchy identify the four terminal
feet.  Each sampled pose is compared with that foot's own rest height, avoiding
false failures when an autoregressive rig places terminal bone heads at slightly
different depths inside otherwise grounded paws.  The evaluated mesh floor is
also measured so a tail, belly, or stretched skin membrane cannot penetrate the
ground unnoticed.
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


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_quadruped_semantics import infer_quadruped_semantics  # noqa: E402


SCHEMA = "avengine_generated_quadruped_foot_contact_audit_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-audit",
        type=Path,
        help=(
            "Optional authenticated source-rig foot-contact audit. Candidate "
            "mesh-floor penetration is then rejected only when it exceeds the "
            "source action baseline by the configured penetration tolerance."
        ),
    )
    parser.add_argument(
        "--front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--contact-tolerance-ratio", type=float, default=0.025)
    parser.add_argument("--penetration-tolerance-ratio", type=float, default=0.015)
    parser.add_argument("--airborne-tolerance-ratio", type=float, default=0.03)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_audit(path):
    if path is None:
        return None, {}
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe reference audit: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid reference audit: {error}") from error
    if (
        payload.get("schema") != SCHEMA
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise SystemExit("reference audit contract is invalid")
    diagonal = float(payload.get("rest", {}).get("mesh_diagonal", 0.0))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise SystemExit("reference audit has invalid rest mesh diagonal")
    actions = {}
    for record in payload.get("actions", []):
        name = record.get("requested_action")
        value = record.get("summary", {}).get("minimum_foot_or_mesh_floor_delta")
        if not isinstance(name, str) or name in actions:
            raise SystemExit("reference audit actions are missing or ambiguous")
        ratio = float(value) / diagonal
        if not math.isfinite(ratio) or ratio < -0.05 or ratio > 0.05:
            raise SystemExit("reference audit floor baseline ratio is outside [-0.05, 0.05]")
        actions[name.lower()] = ratio
    if not actions:
        raise SystemExit("reference audit contains no action baselines")
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "input": payload.get("input"),
        "normalized_minimum_foot_or_mesh_floor_delta_by_action": actions,
    }
    return record, actions


def hidden_objects():
    collection = bpy.data.collections.get("glTF_not_exported")
    return set(collection.objects) if collection is not None else set()


def linked_armatures(mesh):
    result = set()
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        result.add(mesh.parent)
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            result.add(modifier.object)
    return result


def evaluated_vertices(mesh_object, depsgraph):
    evaluated = mesh_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float64)))
        return (homogeneous @ matrix.T)[:, :3]
    finally:
        evaluated.to_mesh_clear()


def weighted_skin_bone_names(mesh, armature, minimum_weight=1.0e-8):
    """Return deform bones plus their ancestors, excluding detached IK controls."""
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    weighted = set()
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            if membership.weight <= minimum_weight:
                continue
            name = group_names.get(membership.group)
            if name is not None and armature.data.bones.get(name) is not None:
                weighted.add(name)
    if not weighted:
        raise SystemExit("skinned mesh has no positive-weight armature vertex groups")

    included = set(weighted)
    for name in tuple(weighted):
        bone = armature.data.bones.get(name)
        while bone is not None:
            included.add(bone.name)
            bone = bone.parent
    return included


def target_records(armature, include_names=None):
    include_names = (
        set(include_names)
        if include_names is not None
        else {bone.name for bone in armature.data.bones}
    )
    records = []
    for bone in armature.data.bones:
        if bone.name not in include_names:
            continue
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        records.append(
            {
                "name": bone.name,
                "parent": (
                    bone.parent.name
                    if bone.parent is not None and bone.parent.name in include_names
                    else None
                ),
                "children": [
                    child.name for child in bone.children if child.name in include_names
                ],
                "head_world": [float(value) for value in head],
                "tail_world": [float(value) for value in tail],
            }
        )
    return records


def resolve_action(name, available):
    matches = [item for item in available if name.lower() in item.name.lower()]
    if len(matches) != 1:
        raise SystemExit(
            f"action {name} is missing or ambiguous: {[item.name for item in matches]}"
        )
    return matches[0]


def main():
    args = parse_argv()
    source = args.input.absolute()
    output = args.output.absolute()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 4 <= args.samples <= 120:
        raise SystemExit("--samples must be in [4, 120]")
    for value, label in (
        (args.contact_tolerance_ratio, "contact tolerance"),
        (args.penetration_tolerance_ratio, "penetration tolerance"),
        (args.airborne_tolerance_ratio, "airborne tolerance"),
    ):
        if not 0.0 < value < 0.2:
            raise SystemExit(f"{label} must be in (0, 0.2)")

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
    body = meshes[0]
    armature = armatures[0]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    armature.animation_data_clear()
    armature.data.pose_position = "REST"
    scene.frame_set(1)
    bpy.context.view_layer.update()
    rest_vertices = evaluated_vertices(body, depsgraph)
    bbox_min = rest_vertices.min(axis=0)
    bbox_extent = np.ptp(rest_vertices, axis=0)
    diagonal = float(np.linalg.norm(bbox_extent))
    if diagonal <= 0.0:
        raise SystemExit("rest mesh has zero diagonal")
    semantic_bones = weighted_skin_bone_names(body, armature)
    excluded_control_bones = sorted(
        bone.name for bone in armature.data.bones if bone.name not in semantic_bones
    )
    semantics = infer_quadruped_semantics(
        target_records(armature, semantic_bones),
        bbox_min=bbox_min,
        bbox_extent=bbox_extent,
        front_axis=args.front_axis,
    )
    rest_foot_z = {
        name: float((armature.matrix_world @ armature.data.bones[name].head_local).z)
        for name in semantics.foot_leaves
    }
    rest_mesh_floor = float(rest_vertices[:, 2].min())
    thresholds = {
        "contact_tolerance_ratio": float(args.contact_tolerance_ratio),
        "penetration_tolerance_ratio": float(args.penetration_tolerance_ratio),
        "airborne_tolerance_ratio": float(args.airborne_tolerance_ratio),
        "contact_tolerance_world": diagonal * args.contact_tolerance_ratio,
        "penetration_tolerance_world": diagonal * args.penetration_tolerance_ratio,
        "airborne_tolerance_world": diagonal * args.airborne_tolerance_ratio,
    }
    reference_record, reference_ratios = load_reference_audit(args.reference_audit)

    records = []
    available = list(bpy.data.actions)
    for requested in args.action or ["Walking", "Idle"]:
        action = resolve_action(requested, available)
        armature.data.pose_position = "POSE"
        armature.animation_data_create()
        armature.animation_data.action = action
        start, end = action.frame_range
        frames = []
        for value in np.linspace(start, end, args.samples):
            evaluated_frame = int(round(float(value)))
            scene.frame_set(evaluated_frame)
            bpy.context.view_layer.update()
            deltas = {}
            for name in semantics.foot_leaves:
                posed_head = armature.matrix_world @ armature.pose.bones[name].head
                deltas[name] = float(posed_head.z - rest_foot_z[name])
            vertices = evaluated_vertices(body, depsgraph)
            mesh_floor_delta = float(vertices[:, 2].min() - rest_mesh_floor)
            contact = [
                name
                for name, delta in deltas.items()
                if abs(delta) <= thresholds["contact_tolerance_world"]
            ]
            frames.append(
                {
                    "source_frame": float(value),
                    "evaluated_frame": evaluated_frame,
                    "foot_vertical_delta_from_own_rest": deltas,
                    "minimum_foot_delta": min(deltas.values()),
                    "maximum_foot_delta": max(deltas.values()),
                    "contact_feet": sorted(contact),
                    "evaluated_mesh_floor_delta_from_rest": mesh_floor_delta,
                }
            )
        worst_penetration = min(
            min(frame["minimum_foot_delta"], frame["evaluated_mesh_floor_delta_from_rest"])
            for frame in frames
        )
        worst_all_airborne = max(frame["minimum_foot_delta"] for frame in frames)
        no_contact_frames = sum(not frame["contact_feet"] for frame in frames)
        candidate_penetration_ratio = worst_penetration / diagonal
        reference_ratio = reference_ratios.get(requested.lower())
        reference_rejection_limit_ratio = (
            reference_ratio - args.penetration_tolerance_ratio
            if reference_ratio is not None
            else None
        )
        if (
            reference_rejection_limit_ratio is not None
            and candidate_penetration_ratio < reference_rejection_limit_ratio
        ):
            decision = "rejected_excess_ground_penetration_beyond_reference"
        elif (
            reference_rejection_limit_ratio is None
            and worst_penetration < -thresholds["penetration_tolerance_world"]
        ):
            decision = "rejected_ground_penetration"
        elif worst_all_airborne > thresholds["airborne_tolerance_world"]:
            decision = "rejected_all_feet_airborne"
        elif no_contact_frames:
            decision = "manual_review_intermittent_contact_proxy"
        else:
            decision = (
                "passed_reference_calibrated_foot_contact_proxy"
                if reference_rejection_limit_ratio is not None
                else "passed_foot_contact_proxy"
            )
        records.append(
            {
                "requested_action": requested,
                "resolved_action": action.name,
                "frame_range": [float(start), float(end)],
                "sampled_frames": frames,
                "summary": {
                    "minimum_foot_or_mesh_floor_delta": worst_penetration,
                    "minimum_foot_or_mesh_floor_delta_ratio": (
                        candidate_penetration_ratio
                    ),
                    "reference_baseline_ratio": reference_ratio,
                    "reference_rejection_limit_ratio": (
                        reference_rejection_limit_ratio
                    ),
                    "maximum_all_feet_airborne_delta": worst_all_airborne,
                    "frames_without_contact_proxy": int(no_contact_frames),
                },
                "decision": decision,
            }
        )

    if any(item["decision"].startswith("rejected_") for item in records):
        overall = "rejected"
    elif any(item["decision"].startswith("manual_") for item in records):
        overall = "manual_review_required"
    else:
        overall = "passed"
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "coordinate_contract": {
            "up_axis": "positive-z",
            "reviewed_front_axis": args.front_axis,
            "automatic_fine_yaw": False,
        },
        "semantics": {
            "root": semantics.root,
            "foot_leaves": list(semantics.foot_leaves),
            "bone_name_independent": True,
            "included_skin_bones_and_ancestors": sorted(semantic_bones),
            "excluded_non_skin_control_bones": excluded_control_bones,
        },
        "rest": {
            "mesh_diagonal": diagonal,
            "mesh_floor": rest_mesh_floor,
            "foot_head_z": rest_foot_z,
        },
        "thresholds": thresholds,
        "reference_audit": reference_record,
        "actions": records,
        "overall": overall,
        "formal_dataset_registration_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"GENERATED_QUADRUPED_FOOT_CONTACT_AUDIT_OK overall={overall} output={output}")


if __name__ == "__main__":
    main()
