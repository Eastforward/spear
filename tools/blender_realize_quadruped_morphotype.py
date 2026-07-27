#!/usr/bin/env python3
"""Freeze a bounded quadruped morphotype guide into a rigged GLB.

The operation is taxonomy-independent and entirely controlled by the
authenticated ``avengine_quadruped_morphotype_guide_v1`` profile.  It infers
the four limbs, axial chain, head chain, and tail from geometry plus skin
structure, realizes the validated target as a new armature rest pose, and
bakes the matching visible geometry into the unchanged mesh topology and skin
weights.  Native Idle/Walking curves are preserved byte-for-byte at the
Blender action-contract level and exported on the realized skeleton.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile

import bpy
import mathutils


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_build_stable_quadruped_instance as stable  # noqa: E402
from tools import blender_render_glb_animation as review  # noqa: E402
from tools.quadruped_morphotype_guide import (  # noqa: E402
    MORPHOTYPE_GUIDE_SCHEMA,
    MorphotypeGuideError,
    build_morphotype_guide_plan,
    load_morphotype_guide_profile,
)


SCHEMA = "avengine_quadruped_morphotype_realization_v1"
MAX_FREEZE_RESIDUAL_HEIGHT_RATIO = 2.0e-6
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-license-id", default="unknown")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_authenticated_input(
    path: Path,
    expected_sha256: str,
    label: str,
    staged_path: Path,
) -> tuple[Path, int]:
    """Authenticate one open descriptor and stage exactly those consumed bytes."""

    path = path.absolute()
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise RuntimeError(f"invalid authenticated {label} sha256")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            f"missing, unsafe, or invalid authenticated {label}: {path}"
        ) from error
    digest = hashlib.sha256()
    copied_size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise RuntimeError(
                f"missing, unsafe, or invalid authenticated {label}: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            with staged_path.open("xb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    destination.write(chunk)
                    copied_size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} sha256 mismatch: expected={expected_sha256} actual={actual}"
        )
    if copied_size != metadata.st_size:
        raise RuntimeError(
            f"{label} size changed while staging: "
            f"opened={metadata.st_size} copied={copied_size}"
        )
    os.chmod(staged_path, 0o400)
    return path, copied_size


def require_new_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_candidate_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,127}", value):
        raise RuntimeError(
            "candidate id must contain 3-128 lowercase ASCII letters, digits, "
            "dot, underscore, or hyphen"
        )
    return value


def exact_canonical_actions(armature):
    actions = list(bpy.data.actions)
    if len(actions) != 2:
        raise RuntimeError(
            "morphotype source must contain exactly Idle and Walking actions; "
            f"available={[item.name for item in actions]}"
        )
    idle = [item for item in actions if item.name.lower().startswith("idle")]
    walking = [
        item
        for item in actions
        if item.name.lower().startswith(("walk", "walking"))
    ]
    if len(idle) != 1 or len(walking) != 1 or idle[0] is walking[0]:
        raise RuntimeError(
            "morphotype source does not have one unambiguous Idle and Walking action"
        )
    idle[0].name = "Idle"
    walking[0].name = "Walking"
    if idle[0].name != "Idle" or walking[0].name != "Walking":
        raise RuntimeError("cannot install canonical Idle/Walking action names")
    clear_animation_state(armature)
    return [idle[0], walking[0]]


def clear_animation_state(armature):
    armature.animation_data_create()
    armature.animation_data.action = None
    for track in list(armature.animation_data.nla_tracks):
        armature.animation_data.nla_tracks.remove(track)
    armature.data.pose_position = "POSE"
    review.reset_pose_basis(armature)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def action_contract_sha256(actions) -> str:
    """Hash complete FCurve/keyframe semantics, not only key coordinates."""

    payload = []
    for action in sorted(actions, key=lambda item: item.name):
        curves = []
        for curve in sorted(
            action.fcurves,
            key=lambda item: (item.data_path, item.array_index),
        ):
            curves.append(
                {
                    "data_path": curve.data_path,
                    "array_index": int(curve.array_index),
                    "extrapolation": curve.extrapolation,
                    "keyframes": [
                        {
                            "co": list(map(float, point.co)),
                            "handle_left": list(map(float, point.handle_left)),
                            "handle_right": list(map(float, point.handle_right)),
                            "handle_left_type": point.handle_left_type,
                            "handle_right_type": point.handle_right_type,
                            "interpolation": point.interpolation,
                            "easing": point.easing,
                        }
                        for point in curve.keyframe_points
                    ],
                }
            )
        payload.append(
            {
                "name": action.name,
                "frame_range": list(map(float, action.frame_range)),
                "curves": curves,
            }
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def skeleton_hierarchy_sha256(armature) -> str:
    digest = hashlib.sha256()
    for bone in sorted(armature.data.bones, key=lambda item: item.name):
        record = (
            bone.name,
            bone.parent.name if bone.parent is not None else "",
            tuple(sorted(child.name for child in bone.children)),
            bool(bone.use_deform),
        )
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def rest_skeleton_sha256(armature) -> str:
    digest = hashlib.sha256()
    for bone in sorted(armature.data.bones, key=lambda item: item.name):
        digest.update(bone.name.encode("utf-8"))
        digest.update(
            struct.pack(
                "<24d",
                *map(float, bone.head_local),
                *map(float, bone.tail_local),
                *(
                    float(bone.matrix_local[row][column])
                    for row in range(4)
                    for column in range(4)
                ),
                float(bone.length),
                float(bone.envelope_distance),
            )
        )
        digest.update(str(bone.inherit_scale).encode("ascii"))
    return digest.hexdigest()


def linked_skinned_meshes(meshes, armature):
    result = []
    for mesh in meshes:
        linked = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object == armature
        ]
        if not linked:
            continue
        active_modifiers = [
            modifier for modifier in mesh.modifiers if modifier.show_viewport
        ]
        if len(linked) != 1 or active_modifiers != linked:
            raise RuntimeError(
                f"skinned mesh {mesh.name!r} must have exactly one active "
                "armature modifier so vertex indexing is preserved"
            )
        if mesh.data.shape_keys is not None:
            raise RuntimeError(
                f"skinned mesh {mesh.name!r} has shape keys; bounded freeze is ambiguous"
            )
        if not mesh.vertex_groups:
            raise RuntimeError(f"skinned mesh {mesh.name!r} has no skin groups")
        result.append(mesh)
    if not result:
        raise RuntimeError("input has no mesh skinned to its single armature")
    return result


def exportable_meshes(meshes):
    """Exclude importer-only custom bone shapes that glTF itself never exports."""

    return [
        mesh
        for mesh in meshes
        if not any(
            collection.name == "glTF_not_exported"
            for collection in mesh.users_collection
        )
    ]


def evaluated_local_positions(mesh):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(
        preserve_all_data_layers=False,
        depsgraph=depsgraph,
    )
    try:
        if len(evaluated_mesh.vertices) != len(mesh.data.vertices):
            raise RuntimeError(
                f"evaluated vertex indexing changed for {mesh.name!r}"
            )
        world_to_local = mesh.matrix_world.inverted()
        return [
            world_to_local @ (evaluated.matrix_world @ vertex.co)
            for vertex in evaluated_mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()


def freeze_pose_as_rest(armature, skinned_meshes):
    target_positions = {
        mesh.name: evaluated_local_positions(mesh) for mesh in skinned_meshes
    }
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    result = bpy.ops.pose.armature_apply(selected=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender could not apply morphotype rest pose: {result}")
    bpy.ops.object.mode_set(mode="OBJECT")
    for mesh in skinned_meshes:
        positions = target_positions[mesh.name]
        flat = [component for point in positions for component in point]
        mesh.data.vertices.foreach_set("co", flat)
        mesh.data.update()
    clear_animation_state(armature)
    return target_positions


def maximum_freeze_residual(skinned_meshes, target_positions) -> float:
    residual = 0.0
    for mesh in skinned_meshes:
        current = evaluated_local_positions(mesh)
        target = target_positions[mesh.name]
        residual = max(
            residual,
            max((left - right).length for left, right in zip(current, target)),
        )
    return residual


def maximum_rest_target_residual(armature, targets) -> float:
    residual = 0.0
    for name, target in targets.items():
        bone = armature.data.bones.get(name)
        if bone is None:
            raise RuntimeError(f"realized skeleton lost target bone {name!r}")
        actual_head = armature.matrix_world @ bone.head_local
        actual_tail = armature.matrix_world @ bone.tail_local
        residual = max(
            residual,
            (actual_head - mathutils.Vector(target.head_world)).length,
            (actual_tail - mathutils.Vector(target.tail_world)).length,
        )
    return residual


def rest_chain_length(armature, names) -> float:
    return sum(
        (
            (armature.matrix_world @ armature.data.bones[name].tail_local)
            - (armature.matrix_world @ armature.data.bones[name].head_local)
        ).length
        for name in names
    )


def configure_walk_idle_export(armature, actions):
    clear_animation_state(armature)
    for action in actions:
        start, end = map(float, action.frame_range)
        track = armature.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, int(round(start)), action)
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_selected_instance(output_path, armature, meshes):
    bpy.ops.object.select_all(action="DESELECT")
    for item in [armature, *meshes]:
        item.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_nla_strips=True,
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender could not export realized GLB: {result}")


def read_exported_glb_contract(
    path: Path,
    *,
    expected_mesh_count: int,
    expected_joint_count: int,
) -> dict:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise RuntimeError("exported GLB has a truncated header")
        magic, version, declared_size = struct.unpack("<III", header)
        if magic != GLB_MAGIC or version != 2 or declared_size != path.stat().st_size:
            raise RuntimeError("exported GLB has an invalid container header")
        document = None
        consumed = 12
        while consumed < declared_size:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise RuntimeError("exported GLB has a truncated chunk header")
            chunk_size, chunk_type = struct.unpack("<II", chunk_header)
            chunk = stream.read(chunk_size)
            if len(chunk) != chunk_size:
                raise RuntimeError("exported GLB has a truncated chunk")
            consumed += 8 + chunk_size
            if chunk_type == GLB_JSON_CHUNK:
                if document is not None:
                    raise RuntimeError("exported GLB has multiple JSON chunks")
                document = json.loads(chunk.rstrip(b"\x00 \t\r\n"))
        if consumed != declared_size or document is None:
            raise RuntimeError("exported GLB JSON/container length mismatch")

    nodes = document.get("nodes", [])
    mesh_nodes = [node for node in nodes if "mesh" in node]
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    animation_names = [item.get("name") for item in animations]
    if len(mesh_nodes) != expected_mesh_count:
        raise RuntimeError(
            "exported GLB mesh count mismatch: "
            f"expected={expected_mesh_count} actual={len(mesh_nodes)}"
        )
    if any("skin" not in node for node in mesh_nodes):
        raise RuntimeError("exported GLB contains an unskinned mesh")
    if len(skins) != 1 or len(skins[0].get("joints", [])) != expected_joint_count:
        raise RuntimeError(
            "exported GLB skin/joint readback mismatch: "
            f"skins={len(skins)} joints="
            f"{[len(item.get('joints', [])) for item in skins]}"
        )
    if sorted(animation_names) != ["Idle", "Walking"]:
        raise RuntimeError(
            f"exported GLB action readback mismatch: {animation_names}"
        )
    channel_counts = {
        item["name"]: len(item.get("channels", []))
        for item in animations
    }
    if any(count <= 0 for count in channel_counts.values()):
        raise RuntimeError(
            f"exported GLB contains an empty animation: {channel_counts}"
        )
    return {
        "passed": True,
        "mesh_count": len(mesh_nodes),
        "all_meshes_skinned": True,
        "skin_count": len(skins),
        "joint_count": len(skins[0]["joints"]),
        "animation_names": animation_names,
        "animation_channel_counts": channel_counts,
    }


def main():
    args = parse_argv()
    candidate_id = validate_candidate_id(args.candidate_id)
    output_path = require_new_output(args.output_glb, "output GLB")
    manifest_path = require_new_output(args.manifest, "manifest")
    if output_path == manifest_path:
        raise RuntimeError("output GLB and manifest must be different paths")
    staging_context = tempfile.TemporaryDirectory(
        prefix="avengine-quadruped-morphotype-"
    )
    staging_root = Path(staging_context.name)
    input_original_path, source_size = stage_authenticated_input(
        args.input,
        args.input_sha256,
        "source GLB",
        staging_root / "source.glb",
    )
    profile_original_path, profile_size = stage_authenticated_input(
        args.profile,
        args.profile_sha256,
        "morphotype profile",
        staging_root / "profile.json",
    )
    input_path = staging_root / "source.glb"
    profile_path = staging_root / "profile.json"
    try:
        profile = load_morphotype_guide_profile(profile_path)
    except MorphotypeGuideError as error:
        raise RuntimeError(f"invalid morphotype profile: {error}") from error

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise RuntimeError("input must contain exactly one armature and a mesh")
    armature = armatures[0]
    meshes = exportable_meshes(meshes)
    skinned_meshes = linked_skinned_meshes(meshes, armature)
    unskinned = sorted(set(meshes) - set(skinned_meshes), key=lambda item: item.name)
    if unskinned:
        raise RuntimeError(
            "morphotype realization refuses exportable unskinned meshes: "
            f"{[item.name for item in unskinned]}"
        )
    body = max(skinned_meshes, key=lambda item: len(item.data.vertices))
    actions = exact_canonical_actions(armature)

    topology_skin_before = stable.mesh_contract_sha256(meshes)
    vertex_before = stable.vertex_position_sha256(meshes)
    action_before = action_contract_sha256(actions)
    hierarchy_before = skeleton_hierarchy_sha256(armature)
    rest_skeleton_before = rest_skeleton_sha256(armature)
    source_minimum, source_maximum = review.evaluated_mesh_bbox(skinned_meshes)
    semantics, records, _minimum, _maximum, extent = (
        review.infer_canonical_quadruped(armature, body)
    )
    try:
        plan = build_morphotype_guide_plan(
            profile,
            semantics,
            records,
            bbox_height=extent[2],
        )
    except MorphotypeGuideError as error:
        raise RuntimeError(f"morphotype guide rejected source rig: {error}") from error

    staging_diagnostics = review.apply_quadruped_morphotype_guide(
        armature,
        body,
        profile,
    )
    target_positions = freeze_pose_as_rest(armature, skinned_meshes)
    height = max(float(extent[2]), 1.0e-12)
    freeze_residual = maximum_freeze_residual(
        skinned_meshes, target_positions
    )
    rest_target_residual = maximum_rest_target_residual(
        armature, plan.targets
    )
    realized_tail_ratio = (
        rest_chain_length(armature, plan.tail_chain)
        / plan.source_tail_length
    )
    realized_minimum, realized_maximum = review.evaluated_mesh_bbox(
        skinned_meshes
    )
    mesh_ground_residual = abs(realized_minimum[2] - source_minimum[2])

    topology_skin_after = stable.mesh_contract_sha256(meshes)
    vertex_after = stable.vertex_position_sha256(meshes)
    action_after = action_contract_sha256(actions)
    hierarchy_after = skeleton_hierarchy_sha256(armature)
    rest_skeleton_after = rest_skeleton_sha256(armature)
    if topology_skin_before != topology_skin_after:
        raise RuntimeError("realization changed topology, UVs, or skin weights")
    if action_before != action_after:
        raise RuntimeError("realization changed Idle/Walking action curves")
    if hierarchy_before != hierarchy_after:
        raise RuntimeError("realization changed skeleton hierarchy")
    if vertex_before == vertex_after or rest_skeleton_before == rest_skeleton_after:
        raise RuntimeError("morphotype realization did not change geometry and rest pose")
    ratio_checks = {
        "freeze_geometry_residual_height_ratio": freeze_residual / height,
        "rest_bone_target_residual_height_ratio": rest_target_residual / height,
        "mesh_ground_residual_height_ratio": mesh_ground_residual / height,
        "tail_length_ratio_error": abs(
            realized_tail_ratio - profile.tail_length_ratio
        ),
    }
    for label, value in ratio_checks.items():
        limit = (
            profile.maximum_ground_residual_height_ratio
            if label == "mesh_ground_residual_height_ratio"
            else MAX_FREEZE_RESIDUAL_HEIGHT_RATIO
        )
        if not math.isfinite(value) or value > limit:
            raise RuntimeError(
                f"{label}={value:.9g} exceeds realization limit {limit:.9g}"
            )

    configure_walk_idle_export(armature, actions)
    export_selected_instance(output_path, armature, meshes)
    glb_readback = read_exported_glb_contract(
        output_path,
        expected_mesh_count=len(meshes),
        expected_joint_count=len(armature.data.bones),
    )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "authority_constraints": {
            "preserves_source_mesh_identity": True,
            "may_serve_as_pose_guide": True,
            "new_taxon_geometry_substitution_authorized": False,
        },
        "source": {
            "path": str(input_original_path),
            "sha256": args.input_sha256,
            "size_bytes": source_size,
            "license_id": args.source_license_id,
            "authenticated_bytes_staged_before_blender_import": True,
        },
        "profile": {
            "path": str(profile_original_path),
            "sha256": args.profile_sha256,
            "size_bytes": profile_size,
            "schema": MORPHOTYPE_GUIDE_SCHEMA,
            "authenticated_bytes_staged_before_parse": True,
            "transforms": {
                "leg_length_ratio": profile.leg_length_ratio,
                "tail_length_ratio": profile.tail_length_ratio,
            },
            "validation": {
                "maximum_ground_residual_height_ratio": (
                    profile.maximum_ground_residual_height_ratio
                )
            },
        },
        "realization": {
            "mode": "validated_pose_to_rest_skeleton_and_base_geometry_v1",
            "semantic_authority_mesh": body.name,
            "skinned_mesh_count": len(skinned_meshes),
            "mesh_count": len(meshes),
            "limb_chain_lengths": {
                label: len(chain)
                for label, chain in plan.limb_chains.items()
            },
            "tail_chain_length": len(plan.tail_chain),
            "topology_uv_skin_sha256_before": topology_skin_before,
            "topology_uv_skin_sha256_after": topology_skin_after,
            "topology_uv_skin_unchanged": True,
            "vertex_position_sha256_before": vertex_before,
            "vertex_position_sha256_after": vertex_after,
            "vertex_positions_changed": True,
            "skeleton_hierarchy_sha256_before": hierarchy_before,
            "skeleton_hierarchy_sha256_after": hierarchy_after,
            "skeleton_hierarchy_unchanged": True,
            "rest_skeleton_sha256_before": rest_skeleton_before,
            "rest_skeleton_sha256_after": rest_skeleton_after,
            "rest_skeleton_changed": True,
            "idle_walking_action_sha256_before": action_before,
            "idle_walking_action_sha256_after": action_after,
            "blender_action_curves_unchanged_before_export": True,
            "actions": ["Idle", "Walking"],
            "exported_glb_readback": glb_readback,
            "realized_tail_length_ratio": realized_tail_ratio,
            **ratio_checks,
            "source_bbox": {
                "minimum": list(map(float, source_minimum)),
                "maximum": list(map(float, source_maximum)),
            },
            "realized_bbox": {
                "minimum": list(map(float, realized_minimum)),
                "maximum": list(map(float, realized_maximum)),
            },
            "staging_guide_diagnostics": {
                key: value
                for key, value in staging_diagnostics.items()
                if key not in {"render_only", "source_asset_unchanged"}
            },
        },
        "artifacts": {
            "glb": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        },
        "qa": {
            "glb_readback": "passed",
            "walking_deformation": "pending",
            "idle_deformation": "pending",
            "walking_visual_review": "pending",
            "idle_visual_review": "pending",
            "ue_asset_bound_readback": "pending",
            "dynamic_audio": "pending",
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging_context.cleanup()
    print(
        "QUADRUPED_MORPHOTYPE_REALIZATION_OK "
        f"candidate={candidate_id} output={output_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
