#!/usr/bin/env python3
"""Attach generated SkinTokens weights to the proven template animation rig.

The generated mesh, PBR material, and SkinTokens vertex weights remain the
asset authority.  The generated SkinTokens skeleton is discarded after its
bone groups are mapped semantically onto the original Quaternius deform bones.
The complete authored Walk/Idle carrier rig and actions are then retained
unchanged, avoiding a second retarget that would drop foot-contact channels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.blender_retarget_quaternius_to_generated_quadruped import (  # noqa: E402
    ACTION_HINTS,
    SIDE_CHAIN_SWAP,
    SOURCE_CHAINS,
    SOURCE_ROOT,
    export_target,
    import_source,
    import_target,
    imported_real_meshes,
    load_motion_basis_decision,
    remove_export_extras,
    require_file,
    require_output,
    sha256_file,
    target_chains,
)


SCHEMA = "avengine_generated_skin_template_animation_graft_v2"
FIXED_DECISION_SCHEMA = "avengine_fixed_skeleton_motion_basis_agent_decision_v1"


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-skinned-glb", type=Path, required=True)
    parser.add_argument("--animated-template-carrier-glb", type=Path, required=True)
    parser.add_argument("--source-motion-glb", type=Path, required=True)
    parser.add_argument("--motion-basis-decision", type=Path, required=True)
    parser.add_argument(
        "--fixed-skeleton-conditioning-manifest",
        type=Path,
        help=(
            "Enable hierarchy-authenticated fixed-skeleton mapping instead of "
            "the legacy AnimalPack semantic-name map."
        ),
    )
    parser.add_argument("--skintokens-attempt", type=Path)
    parser.add_argument("--static-rig-audit", type=Path)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--target-front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument(
        "--maximum-carrier-bbox-error-ratio",
        type=float,
        default=0.03,
        help="Fail when the carrier mesh and generated weighted mesh are not aligned.",
    )
    parser.add_argument(
        "--maximum-rest-bind-error-ratio",
        type=float,
        default=1.0e-5,
        help="Fail when changing armatures moves any generated vertex in rest pose.",
    )
    parser.add_argument(
        "--minimum-action-deformation-ratio",
        type=float,
        default=1.0e-5,
        help="Fail when a named Walk/Idle action exports as an effectively static clip.",
    )
    parser.add_argument(
        "--maximum-hierarchy-segment-error-ratio",
        type=float,
        default=0.30,
        help=(
            "Maximum normalized rest-segment discrepancy when a TokenRig "
            "anonymous bone is mapped through fixed-skeleton hierarchy lineage."
        ),
    )
    return parser.parse_args(argv)


def world_vertex_positions(mesh_object):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(
        preserve_all_data_layers=False,
        depsgraph=depsgraph,
    )
    try:
        matrix = evaluated.matrix_world
        return [matrix @ vertex.co for vertex in evaluated_mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def point_bbox(points):
    if not points:
        raise RuntimeError("cannot measure an empty mesh")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum, maximum - minimum


def bbox_alignment(first_points, second_points):
    first_minimum, first_maximum, first_extent = point_bbox(first_points)
    second_minimum, second_maximum, second_extent = point_bbox(second_points)
    reference = max(first_extent.length, second_extent.length, 1.0e-8)
    center_error = (
        ((first_minimum + first_maximum) * 0.5)
        - ((second_minimum + second_maximum) * 0.5)
    ).length
    extent_error = (first_extent - second_extent).length
    return {
        "generated_bbox_min": [float(value) for value in first_minimum],
        "generated_bbox_max": [float(value) for value in first_maximum],
        "generated_bbox_extent": [float(value) for value in first_extent],
        "carrier_bbox_min": [float(value) for value in second_minimum],
        "carrier_bbox_max": [float(value) for value in second_maximum],
        "carrier_bbox_extent": [float(value) for value in second_extent],
        "center_error_ratio": float(center_error / reference),
        "extent_error_ratio": float(extent_error / reference),
        "maximum_error_ratio": float(max(center_error, extent_error) / reference),
    }


def read_json(path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def canonical_hash_without(value, key):
    body = {name: item for name, item in value.items() if name != key}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def authenticate_record(record, path, label):
    if (
        not isinstance(record, dict)
        or record.get("sha256") != sha256_file(path)
        or int(record.get("size_bytes", -1)) != path.stat().st_size
    ):
        raise RuntimeError(f"{label} identity mismatch")


def load_fixed_lineage_contract(
    *,
    decision_path,
    conditioning_path,
    attempt_path,
    audit_path,
    target_path,
    carrier_path,
    front_axis,
):
    decision = read_json(decision_path, "fixed-skeleton motion decision")
    conditioning = read_json(conditioning_path, "conditioning manifest")
    attempt = read_json(attempt_path, "SkinTokens attempt")
    audit = read_json(audit_path, "static rig audit")
    if (
        decision.get("schema") != FIXED_DECISION_SCHEMA
        or decision.get("status") != "agent_research_approved"
        or decision.get("decision_authority") != "agent_delegated_research"
        or decision.get("human_approved") is not False
        or decision.get("agent_approved_for_research") is not True
        or decision.get("user_delegated_autonomous_review") is not True
        or decision.get("target_animation_generation_authorized") is not True
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("decision_sha256")
        != canonical_hash_without(decision, "decision_sha256")
    ):
        raise RuntimeError("fixed-skeleton decision is not authenticated research approval")
    authenticate_record(decision.get("target"), target_path, "decision target")
    authenticate_record(
        decision.get("animated_template_carrier"), carrier_path, "decision carrier"
    )
    if decision.get("target", {}).get("reviewed_front_axis") != front_axis:
        raise RuntimeError("fixed-skeleton decision target front axis mismatch")
    if (
        int(decision.get("manual_cardinal_motion_basis_yaw_deg", 999)) != 0
        or decision.get("side_chain_mode") != "matched"
        or decision.get("rotation_transfer_mode") != "unchanged_template_actions"
    ):
        raise RuntimeError("fixed-skeleton decision changed the proven zero-yaw basis")

    lineage = decision.get("authenticated_lineage", {})
    authenticate_record(
        lineage.get("conditioning_manifest"),
        conditioning_path,
        "decision conditioning manifest",
    )
    authenticate_record(
        lineage.get("skintokens_attempt"), attempt_path, "decision SkinTokens attempt"
    )
    authenticate_record(
        lineage.get("static_rig_audit"), audit_path, "decision static rig audit"
    )
    if (
        conditioning.get("schema")
        != "avengine_fixed_quadruped_skeleton_conditioning_v3"
        or conditioning.get("formal_dataset_registration_authorized") is not False
        or attempt.get("schema") != "avengine_fixed_skeleton_skintokens_attempt_v1"
        or attempt.get("status") != "succeeded"
        or attempt.get("formal_dataset_registration_authorized") is not False
        or audit.get("schema") != "avengine_generated_animal_rig_audit_v1"
        or audit.get("automatic_checks", {}).get("overall") != "passed"
        or audit.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("fixed-skeleton lineage contract is incomplete")
    authenticate_record(conditioning.get("input"), carrier_path, "conditioning carrier")
    conditioning_output = require_file(
        Path(str(conditioning.get("output", {}).get("path", ""))),
        "conditioning output",
    )
    authenticate_record(
        conditioning.get("output"), conditioning_output, "conditioning output"
    )
    authenticate_record(attempt.get("input"), conditioning_output, "SkinTokens input")
    authenticate_record(attempt.get("output"), target_path, "SkinTokens output")
    authenticate_record(audit.get("input"), target_path, "static audit target")
    retained = conditioning.get("skeleton_conditioning", {}).get("retained_bones")
    reparented = conditioning.get("skeleton_conditioning", {}).get(
        "reparented_weight_bearing_roots"
    )
    root = conditioning.get("skeleton_conditioning", {}).get("root_after")
    if (
        not isinstance(retained, list)
        or not retained
        or len(retained) != len(set(retained))
        or not isinstance(reparented, list)
        or root not in retained
    ):
        raise RuntimeError("conditioning manifest has invalid retained hierarchy")
    return decision, conditioning, attempt, audit


def import_animated_carrier(path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = [item for item in bpy.data.actions if item not in before_actions]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("animated carrier must contain exactly one armature")
    selected = {}
    for canonical, hint in ACTION_HINTS.items():
        matches = [item for item in actions if hint in item.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(
                f"carrier action {canonical} is ambiguous: {[item.name for item in matches]}"
            )
        selected[canonical] = matches[0]
    return armatures[0], imported, selected


def bone_segment_world(armature, name):
    bone = armature.data.bones[name]
    return armature.matrix_world @ bone.head_local, armature.matrix_world @ bone.tail_local


def hierarchy_vertex_group_mapping(
    target,
    mesh,
    carrier,
    conditioning,
    maximum_error_ratio,
):
    contract = conditioning["skeleton_conditioning"]
    retained = set(contract["retained_bones"])
    root_name = contract["root_after"]
    carrier_names = set(carrier.data.bones.keys())
    if not retained.issubset(carrier_names):
        raise RuntimeError(
            "animated carrier lost conditioning bones: "
            f"{sorted(retained - carrier_names)}"
        )
    target_roots = [bone.name for bone in target.data.bones if bone.parent is None]
    if len(target_roots) != 1:
        raise RuntimeError(f"fixed SkinTokens target must have one root: {target_roots}")

    logical_parent = {}
    for name in retained:
        bone = carrier.data.bones[name]
        logical_parent[name] = (
            bone.parent.name if bone.parent is not None and bone.parent.name in retained else None
        )
    for record in contract["reparented_weight_bearing_roots"]:
        name = record.get("bone")
        parent = record.get("parent_after")
        if name not in retained or parent not in retained:
            raise RuntimeError("conditioning root-parent record escaped retained skeleton")
        logical_parent[name] = parent
    logical_children = {name: [] for name in retained}
    for name, parent in logical_parent.items():
        if parent is not None:
            logical_children[parent].append(name)
    for names in logical_children.values():
        names.sort()

    _minimum, _maximum, extent = point_bbox(world_vertex_positions(mesh))
    reference = max(extent.length, 1.0e-8)

    def pair_error(target_name, carrier_name):
        target_head, target_tail = bone_segment_world(target, target_name)
        carrier_head, carrier_tail = bone_segment_world(carrier, carrier_name)
        return float(
            ((target_head - carrier_head).length + (target_tail - carrier_tail).length)
            / reference
        )

    mapping = {target_roots[0]: root_name}
    queue = [target_roots[0]]
    while queue:
        target_parent = queue.pop(0)
        carrier_parent = mapping[target_parent]
        target_children = sorted(child.name for child in target.data.bones[target_parent].children)
        carrier_children = logical_children[carrier_parent]
        if len(target_children) > len(carrier_children):
            raise RuntimeError(
                "fixed hierarchy has more target children than carrier candidates: "
                f"target={target_parent}/{target_children} "
                f"carrier={carrier_parent}/{carrier_children}"
            )
        if not target_children:
            continue
        assignments = itertools.permutations(carrier_children, len(target_children))
        chosen = min(
            assignments,
            key=lambda values: sum(
                pair_error(target_name, carrier_name)
                for target_name, carrier_name in zip(target_children, values)
            ),
        )
        for target_name, carrier_name in zip(target_children, chosen):
            mapping[target_name] = carrier_name
            queue.append(target_name)

    target_names = set(target.data.bones.keys())
    if set(mapping) != target_names or len(set(mapping.values())) != len(mapping):
        raise RuntimeError(
            "fixed hierarchy mapping is incomplete or non-injective: "
            f"missing={sorted(target_names - set(mapping))}"
        )
    records = []
    for target_name in sorted(mapping):
        carrier_name = mapping[target_name]
        error = pair_error(target_name, carrier_name)
        target_parent = target.data.bones[target_name].parent
        records.append(
            {
                "target_bone": target_name,
                "template_bone": carrier_name,
                "target_parent": target_parent.name if target_parent is not None else None,
                "template_logical_parent": logical_parent[carrier_name],
                "rest_segment_error_ratio": error,
            }
        )
    maximum = max(record["rest_segment_error_ratio"] for record in records)
    if maximum > maximum_error_ratio:
        raise RuntimeError(
            "fixed hierarchy rest-segment mapping exceeds threshold: "
            f"actual={maximum:.8f} allowed={maximum_error_ratio:.8f}"
        )
    group_names = {group.name for group in mesh.vertex_groups}
    if group_names != target_names:
        raise RuntimeError(
            "generated mesh groups differ from the fixed target skeleton: "
            f"missing={sorted(target_names - group_names)} "
            f"extra={sorted(group_names - target_names)}"
        )
    return mapping, records, {
        "method": "conditioning_hierarchy_child_assignment_v1",
        "target_bones": len(mapping),
        "carrier_retained_bones": len(retained),
        "unused_carrier_bones": sorted(retained - set(mapping.values())),
        "maximum_rest_segment_error_ratio": maximum,
        "allowed_maximum_rest_segment_error_ratio": float(maximum_error_ratio),
        "complete_target_coverage": True,
        "injective_mapping": True,
        "parent_child_lineage_preserved": True,
    }


def semantic_vertex_group_mapping(target, mesh, front_axis, side_chain_mode):
    if side_chain_mode not in {"matched", "swapped"}:
        raise RuntimeError(f"invalid side-chain mode: {side_chain_mode}")
    semantics, chains = target_chains(target, mesh, front_axis)
    mapping = {semantics.root: SOURCE_ROOT}
    records = [
        {"target_bone": semantics.root, "template_bone": SOURCE_ROOT, "chain": "root"}
    ]
    for semantic, target_names in chains.items():
        source_semantic = (
            SIDE_CHAIN_SWAP.get(semantic, semantic)
            if side_chain_mode == "swapped"
            else semantic
        )
        source_names = SOURCE_CHAINS[source_semantic]
        if len(target_names) != len(source_names):
            raise RuntimeError(
                "template animation graft requires equal semantic chain lengths: "
                f"{semantic} target={len(target_names)} source={len(source_names)}"
            )
        for target_name, source_name in zip(target_names, source_names):
            if target_name in mapping:
                raise RuntimeError(f"duplicate target semantic bone: {target_name}")
            mapping[target_name] = source_name
            records.append(
                {
                    "target_bone": target_name,
                    "template_bone": source_name,
                    "chain": semantic,
                    "template_chain": source_semantic,
                }
            )
    target_names = set(target.data.bones.keys())
    if set(mapping) != target_names:
        raise RuntimeError(
            "template animation graft requires complete target bone coverage: "
            f"missing={sorted(target_names - set(mapping))}"
        )
    group_names = {group.name for group in mesh.vertex_groups}
    if group_names != target_names:
        raise RuntimeError(
            "generated mesh vertex groups must exactly match target bones: "
            f"missing={sorted(target_names - group_names)} "
            f"extra={sorted(group_names - target_names)}"
        )
    return semantics, mapping, records


def rename_vertex_groups(mesh, mapping):
    temporary = {}
    for index, (target_name, template_name) in enumerate(sorted(mapping.items())):
        group = mesh.vertex_groups.get(target_name)
        if group is None:
            raise RuntimeError(f"missing generated vertex group: {target_name}")
        temporary_name = f"__avengine_graft_{index:03d}__"
        group.name = temporary_name
        temporary[temporary_name] = template_name
    for temporary_name, template_name in temporary.items():
        if mesh.vertex_groups.get(template_name) is not None:
            raise RuntimeError(f"template vertex-group collision: {template_name}")
        mesh.vertex_groups[temporary_name].name = template_name


def attach_mesh_to_armature(mesh, armature):
    mesh_world = mesh.matrix_world.copy()
    for modifier in list(mesh.modifiers):
        if modifier.type == "ARMATURE":
            mesh.modifiers.remove(modifier)
    mesh.parent = None
    mesh.matrix_world = mesh_world
    modifier = mesh.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_vertex_groups = True
    mesh.parent = armature
    mesh.parent_type = "OBJECT"
    mesh.matrix_parent_inverse = armature.matrix_world.inverted()
    mesh.matrix_world = mesh_world
    bpy.context.view_layer.update()


def action_channel_summary(actions, deform_bone_names=None):
    if deform_bone_names is None:
        deform_bone_names = {SOURCE_ROOT}
        for chain in SOURCE_CHAINS.values():
            deform_bone_names.update(chain)
    else:
        deform_bone_names = set(deform_bone_names)
    result = {}
    for canonical, action in actions.items():
        channels = {"location": 0, "rotation_quaternion": 0, "scale": 0, "other": 0}
        deform_bones = set()
        for curve in action.fcurves:
            matched = False
            for channel in ("location", "rotation_quaternion", "scale"):
                if curve.data_path.endswith(f".{channel}"):
                    channels[channel] += 1
                    matched = True
                    break
            if not matched:
                channels["other"] += 1
            prefix = 'pose.bones["'
            if curve.data_path.startswith(prefix):
                name = curve.data_path[len(prefix) :].split('"', 1)[0]
                if name in deform_bone_names:
                    deform_bones.add(name)
        result[canonical] = {
            "source_action_name": action.name,
            "frame_range": [float(value) for value in action.frame_range],
            "fcurve_channels": channels,
            "animated_deform_bones": sorted(deform_bones),
        }
    return result


def maximum_point_error_ratio(before, after):
    if len(before) != len(after):
        raise RuntimeError(
            f"rest bind changed vertex count: before={len(before)} after={len(after)}"
        )
    _minimum, _maximum, extent = point_bbox(before)
    reference = max(extent.length, 1.0e-8)
    maximum = max((first - second).length for first, second in zip(before, after))
    return float(maximum), float(maximum / reference)


def probe_action_deformation(armature, mesh, actions, rest_points, minimum_ratio):
    _minimum, _maximum, extent = point_bbox(rest_points)
    reference = max(extent.length, 1.0e-8)
    armature.animation_data_create()
    result = {}
    for canonical, action in actions.items():
        armature.animation_data.action = action
        start, end = [float(value) for value in action.frame_range]
        frames = [start + (end - start) * index / 4.0 for index in range(5)]
        maximum = 0.0
        for value in frames:
            base = int(value)
            bpy.context.scene.frame_set(base, subframe=value - base)
            bpy.context.view_layer.update()
            posed = world_vertex_positions(mesh)
            maximum = max(
                maximum,
                max(
                    (rest - current).length
                    for rest, current in zip(rest_points, posed)
                ),
            )
        ratio = maximum / reference
        if ratio <= minimum_ratio:
            raise RuntimeError(
                f"{canonical} is effectively static after skeleton graft: "
                f"maximum deformation ratio={ratio:.10f}"
            )
        result[canonical] = {
            "sampled_frames": frames,
            "maximum_vertex_displacement_world": float(maximum),
            "maximum_vertex_displacement_ratio": float(ratio),
            "minimum_required_ratio": float(minimum_ratio),
            "non_static": True,
        }
    armature.animation_data.action = None
    return result


def write_manifest(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    args = parse_argv()
    if args.maximum_carrier_bbox_error_ratio <= 0.0:
        raise SystemExit("--maximum-carrier-bbox-error-ratio must be positive")
    if args.maximum_rest_bind_error_ratio <= 0.0:
        raise SystemExit("--maximum-rest-bind-error-ratio must be positive")
    if args.minimum_action_deformation_ratio <= 0.0:
        raise SystemExit("--minimum-action-deformation-ratio must be positive")
    if args.maximum_hierarchy_segment_error_ratio <= 0.0:
        raise SystemExit("--maximum-hierarchy-segment-error-ratio must be positive")

    target_path = require_file(args.target_skinned_glb, "generated skinned GLB")
    carrier_path = require_file(
        args.animated_template_carrier_glb,
        "animated template carrier GLB",
    )
    source_motion_path = require_file(args.source_motion_glb, "source motion GLB")
    decision_path = require_file(args.motion_basis_decision, "motion-basis decision")
    fixed_mode = args.fixed_skeleton_conditioning_manifest is not None
    if fixed_mode:
        if args.skintokens_attempt is None or args.static_rig_audit is None:
            raise SystemExit(
                "fixed-skeleton mapping requires --skintokens-attempt and "
                "--static-rig-audit"
            )
        if source_motion_path != carrier_path:
            raise SystemExit(
                "fixed-skeleton graft requires --source-motion-glb to be the "
                "same immutable animated carrier"
            )
        conditioning_path = require_file(
            args.fixed_skeleton_conditioning_manifest,
            "fixed-skeleton conditioning manifest",
        )
        attempt_path = require_file(args.skintokens_attempt, "SkinTokens attempt")
        audit_path = require_file(args.static_rig_audit, "static rig audit")
        decision, conditioning, _attempt, _audit = load_fixed_lineage_contract(
            decision_path=decision_path,
            conditioning_path=conditioning_path,
            attempt_path=attempt_path,
            audit_path=audit_path,
            target_path=target_path,
            carrier_path=carrier_path,
            front_axis=args.target_front_axis,
        )
        approved_yaw = 0
        approved_side_mode = "matched"
        approved_solver = "unchanged_template_actions"
    else:
        conditioning_path = attempt_path = audit_path = None
        decision_path, decision, approved_yaw, approved_side_mode, approved_solver = (
            load_motion_basis_decision(
                decision_path,
                target_path,
                source_motion_path,
                args.target_front_axis,
            )
        )
    if approved_yaw != 0:
        raise RuntimeError(
            "template animation graft currently requires an approved 0-degree basis"
        )
    output = require_output(args.output_glb, "grafted animated output GLB")
    manifest_path = require_output(args.manifest, "graft manifest")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    target, mesh, _target_imported = import_target(target_path)
    target.data.pose_position = "REST"
    bpy.context.view_layer.update()
    generated_rest_points = world_vertex_positions(mesh)
    mapping_audit = None
    if fixed_mode:
        carrier, carrier_imported, actions = import_animated_carrier(carrier_path)
        mapping, mapping_records, mapping_audit = hierarchy_vertex_group_mapping(
            target,
            mesh,
            carrier,
            conditioning,
            args.maximum_hierarchy_segment_error_ratio,
        )
    else:
        _semantics, mapping, mapping_records = semantic_vertex_group_mapping(
            target,
            mesh,
            args.target_front_axis,
            approved_side_mode,
        )
        carrier, carrier_imported, actions = import_source(carrier_path)
    carrier.data.pose_position = "REST"
    carrier_meshes = imported_real_meshes(carrier_imported)
    if len(carrier_meshes) != 1:
        raise RuntimeError(
            "animated template carrier must contain one real mesh: "
            f"{[item.name for item in carrier_meshes]}"
        )
    required_template_bones = set(mapping.values())
    if not required_template_bones.issubset(set(carrier.data.bones.keys())):
        raise RuntimeError("animated template carrier is missing mapped deform bones")
    bpy.context.view_layer.update()
    carrier_rest_points = world_vertex_positions(carrier_meshes[0])
    carrier_alignment = bbox_alignment(generated_rest_points, carrier_rest_points)
    if carrier_alignment["maximum_error_ratio"] > args.maximum_carrier_bbox_error_ratio:
        raise RuntimeError(
            "animated carrier and generated mesh are not aligned: "
            f"error={carrier_alignment['maximum_error_ratio']:.8f}"
        )

    action_summary = action_channel_summary(actions, set(mapping.values()))
    for canonical in ACTION_HINTS:
        actions[canonical].name = canonical
        actions[canonical].use_fake_user = True
    rename_vertex_groups(mesh, mapping)
    attach_mesh_to_armature(mesh, carrier)
    carrier.data.pose_position = "REST"
    bpy.context.view_layer.update()
    grafted_rest_points = world_vertex_positions(mesh)
    rest_bind_error, rest_bind_error_ratio = maximum_point_error_ratio(
        generated_rest_points,
        grafted_rest_points,
    )
    if rest_bind_error_ratio > args.maximum_rest_bind_error_ratio:
        raise RuntimeError(
            "template animation graft changed the generated rest mesh: "
            f"error_ratio={rest_bind_error_ratio:.10f}"
        )

    # The REST mode above is only for the immutable bind check.  glTF export
    # samples evaluated poses, so leaving the armature in REST would publish
    # named but constant Walk/Idle clips.
    carrier.data.pose_position = "POSE"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    action_deformation_probe = probe_action_deformation(
        carrier,
        mesh,
        actions,
        generated_rest_points,
        args.minimum_action_deformation_ratio,
    )
    removed = remove_export_extras(carrier, mesh)
    output_actions = [actions["Walking"], actions["Idle"]]
    canonical_yaw = export_target(
        carrier,
        mesh,
        output_actions,
        output,
        args.target_front_axis,
    )
    if fixed_mode:
        motion_basis_gate = {
            "decision_path": str(decision_path),
            "decision_sha256": decision["decision_sha256"],
            "decision_file_sha256": sha256_file(decision_path),
            "decision_authority": "agent_delegated_research",
            "human_approved": False,
            "human_approved_by": None,
            "agent_approved_for_research": True,
            "user_delegated_autonomous_review": True,
            "target_animation_generation_authorized": True,
            "formal_dataset_registration_authorized": False,
            "approved_motion_basis_yaw_deg": 0,
            "approved_side_chain_mode": "matched",
            "approved_preview_rotation_solver": "unchanged_template_actions",
            "authenticated_lineage": decision["authenticated_lineage"],
        }
        graft_method = "fixed_skeleton_hierarchy_map_to_unchanged_template_rig_v1"
    else:
        motion_basis_gate = {
            "decision_path": str(decision_path),
            "decision_sha256": decision["decision_sha256"],
            "decision_file_sha256": sha256_file(decision_path),
            "preview_sha256": decision["preview_sha256"],
            "decision_authority": "human",
            "human_approved": True,
            "human_approved_by": decision["human_approved_by"],
            "target_animation_generation_authorized": True,
            "formal_dataset_registration_authorized": False,
            "approved_motion_basis_yaw_deg": int(approved_yaw),
            "approved_side_chain_mode": approved_side_mode,
            "approved_preview_rotation_solver": approved_solver,
        }
        graft_method = "semantic_weight_group_map_to_unchanged_template_rig_v1"
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_authority": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "mesh_preserved": True,
            "pbr_preserved": True,
            "skintokens_vertex_weights_preserved": True,
            "generated_skeleton_discarded": True,
        },
        "animated_template_carrier": {
            "path": str(carrier_path),
            "sha256": sha256_file(carrier_path),
            "geometry_used": False,
            "weights_used": False,
            "skeleton_used": True,
            "walk_idle_actions_used": True,
        },
        "source_motion": {
            "path": str(source_motion_path),
            "sha256": sha256_file(source_motion_path),
            "identity_authenticated_by_human_decision": not fixed_mode,
            "identity_authenticated_by_fixed_skeleton_lineage": fixed_mode,
        },
        "motion_basis_gate": motion_basis_gate,
        "graft": {
            "method": graft_method,
            "second_retarget_performed": False,
            "bone_mapping": mapping_records,
            "hierarchy_mapping_audit": mapping_audit,
            "mapped_deform_bones": len(mapping),
            "complete_generated_bone_and_group_coverage": True,
            "carrier_alignment": carrier_alignment,
            "carrier_alignment_threshold_ratio": float(
                args.maximum_carrier_bbox_error_ratio
            ),
            "rest_bind_maximum_world_error": rest_bind_error,
            "rest_bind_maximum_error_ratio": rest_bind_error_ratio,
            "rest_bind_threshold_ratio": float(args.maximum_rest_bind_error_ratio),
            "rest_mesh_preserved": True,
            "export_armature_pose_position": "POSE",
        },
        "actions": action_summary,
        "pre_export_action_deformation_probe": action_deformation_probe,
        "export": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "canonical_front_axis": "positive-x",
            "canonical_yaw_degrees": canonical_yaw,
            "action_names": ["Walking", "Idle"],
            "removed_export_extras": removed,
        },
        "status": "research_candidate_pending_contact_deformation_and_visual_qa",
        "formal_dataset_registration_authorized": False,
    }
    write_manifest(manifest_path, payload)
    print(
        "GENERATED_SKIN_TEMPLATE_ANIMATION_GRAFT_OK "
        f"mapped_bones={len(mapping)} actions=2 output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
