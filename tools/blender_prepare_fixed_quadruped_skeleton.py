#!/usr/bin/env python3
"""Prepare a single-root, animation-free quadruped skeleton for SkinTokens.

The input is a generated-topology mesh that has already been aligned to a
trusted body-plan rig.  Some Quaternius assets also contain detached,
weight-bearing foot/IK components.  TokenRig's ``--use_skeleton`` path
correctly rejects those as a multi-root skeleton, but deleting them would
discard real foot joints and weights.  This tool keeps every bone that carries
positive skin weight plus its ancestors.  When ``--main-root`` is supplied,
any other retained roots are parented beneath that trusted body root without
moving their rest coordinates.  Animation and unrelated scene objects are
removed while the generated mesh, PBR material, UVs, and fitted rest skeleton
are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy


SCHEMA = "avengine_fixed_quadruped_skeleton_conditioning_v3"


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--reference-rig",
        type=Path,
        help=(
            "Trusted animated template GLB whose positive-weight bone names "
            "define the complete deform skeleton. This prevents an imperfect "
            "temporary mesh transfer from pruning a valid limb, ear, or tail bone."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-weight", type=float, default=1.0e-8)
    parser.add_argument(
        "--main-root",
        default="",
        help=(
            "Trusted principal body root. If retained positive-weight bones "
            "span multiple roots, all other retained roots are parented under "
            "this bone without changing armature-space rest coordinates."
        ),
    )
    parser.add_argument(
        "--armature-name",
        default="Armature",
        help=(
            "Normalize both the armature object and data-block name.  SkinTokens' "
            "current use_origin exporter otherwise looks up the requested armature "
            "name before synchronizing the newly-created Blender object."
        ),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {path}")
    if path.suffix.lower() != ".glb":
        raise SystemExit("fixed-skeleton conditioning currently requires GLB input")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def identify_runtime():
    hidden = hidden_objects()
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    meshes = [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and item not in hidden and linked_armatures(item)
    ]
    if len(armatures) != 1 or len(meshes) != 1:
        raise SystemExit(
            "input must contain exactly one armature and one real skinned mesh; "
            f"armatures={[item.name for item in armatures]} "
            f"meshes={[item.name for item in meshes]}"
        )
    armature, mesh = armatures[0], meshes[0]
    if linked_armatures(mesh) != {armature}:
        raise SystemExit("mesh is linked to an unexpected armature")
    return armature, mesh


def positive_weight_bones(mesh, armature, minimum_weight):
    bone_names = {bone.name for bone in armature.data.bones}
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    weighted = set()
    weighted_vertices = 0
    maximum_influences = 0
    for vertex in mesh.data.vertices:
        influences = 0
        for membership in vertex.groups:
            name = group_names.get(membership.group)
            if name in bone_names and float(membership.weight) > minimum_weight:
                weighted.add(name)
                influences += 1
        if influences:
            weighted_vertices += 1
        maximum_influences = max(maximum_influences, influences)
    if weighted_vertices != len(mesh.data.vertices):
        raise SystemExit(
            "conditioning mesh contains unweighted vertices: "
            f"weighted={weighted_vertices} total={len(mesh.data.vertices)}"
        )
    if not weighted:
        raise SystemExit("conditioning mesh has no positive-weight skin bones")
    return weighted, weighted_vertices, maximum_influences


def load_reference_deform_authority(path, minimum_weight):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    armature, mesh = identify_runtime()
    weighted, weighted_vertices, maximum_influences = positive_weight_bones(
        mesh, armature, minimum_weight
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "armature": armature.name,
        "mesh": mesh.name,
        "weighted_bones": sorted(weighted),
        "weighted_vertices": weighted_vertices,
        "maximum_influences": maximum_influences,
        "roots": root_names(armature, include_ancestors(armature, weighted)),
    }


def include_ancestors(armature, names):
    retained = set(names)
    for name in tuple(names):
        bone = armature.data.bones[name]
        while bone.parent is not None:
            bone = bone.parent
            retained.add(bone.name)
    return retained


def root_names(armature, names):
    return sorted(
        name
        for name in names
        if armature.data.bones[name].parent is None
        or armature.data.bones[name].parent.name not in names
    )


def parent_retained_roots_under_main(armature, retained, main_root):
    roots_before = root_names(armature, retained)
    if main_root not in roots_before:
        raise SystemExit(
            "--main-root must name one of the retained skeleton roots; "
            f"requested={main_root!r} roots={roots_before}"
        )
    if len(roots_before) <= 1:
        return []

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    armature.hide_set(False)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    main = armature.data.edit_bones.get(main_root)
    if main is None:
        raise SystemExit(f"missing edit bone for --main-root: {main_root}")

    reparented = []
    for name in roots_before:
        if name == main_root:
            continue
        bone = armature.data.edit_bones.get(name)
        if bone is None:
            raise SystemExit(f"missing retained root edit bone: {name}")
        before_head = tuple(float(value) for value in bone.head)
        before_tail = tuple(float(value) for value in bone.tail)
        bone.parent = main
        bone.use_connect = False
        after_head = tuple(float(value) for value in bone.head)
        after_tail = tuple(float(value) for value in bone.tail)
        if before_head != after_head or before_tail != after_tail:
            raise SystemExit(
                "root parenting changed armature-space rest coordinates: "
                f"bone={name} before={(before_head, before_tail)} "
                f"after={(after_head, after_tail)}"
            )
        reparented.append(
            {
                "bone": name,
                "parent_before": None,
                "parent_after": main_root,
                "use_connect": False,
                "head_armature_space": before_head,
                "tail_armature_space": before_tail,
            }
        )
    bpy.ops.object.mode_set(mode="OBJECT")

    roots_after = root_names(armature, retained)
    if roots_after != [main_root]:
        raise SystemExit(
            "root parenting did not produce the requested single root; "
            f"requested={main_root!r} roots={roots_after}"
        )
    return reparented


def remove_unretained_bones(armature, retained):
    before = [bone.name for bone in armature.data.bones]
    removed = sorted(set(before) - set(retained))
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    armature.hide_set(False)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    for name in removed:
        edit_bone = armature.data.edit_bones.get(name)
        if edit_bone is not None:
            armature.data.edit_bones.remove(edit_bone)
    bpy.ops.object.mode_set(mode="OBJECT")
    actual = {bone.name for bone in armature.data.bones}
    if actual != set(retained):
        raise SystemExit(
            "bone pruning changed the retained skeleton unexpectedly: "
            f"missing={sorted(set(retained) - actual)} extra={sorted(actual - set(retained))}"
        )
    return removed


def remove_unretained_vertex_groups(mesh, retained):
    removed = []
    for group in list(mesh.vertex_groups):
        if group.name not in retained:
            removed.append(group.name)
            mesh.vertex_groups.remove(group)
    return sorted(removed)


def remove_animation_and_extras(armature, mesh):
    armature.animation_data_clear()
    armature.data.pose_position = "REST"
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    retained = {armature, mesh}
    removed = []
    for item in list(bpy.context.scene.objects):
        if item not in retained:
            removed.append({"name": item.name, "type": item.type})
            bpy.data.objects.remove(item, do_unlink=True)
    return sorted(removed, key=lambda record: (record["type"], record["name"]))


def normalize_armature_name(armature, requested_name):
    requested_name = str(requested_name).strip()
    if not requested_name:
        raise SystemExit("armature name must not be empty")

    object_conflict = bpy.data.objects.get(requested_name)
    if object_conflict is not None and object_conflict != armature:
        raise SystemExit(
            "armature object name conflicts with retained runtime object: "
            f"requested={requested_name} existing={object_conflict.name}"
        )
    data_conflict = bpy.data.armatures.get(requested_name)
    if data_conflict is not None and data_conflict != armature.data:
        raise SystemExit(
            "armature data name conflicts with retained runtime data-block: "
            f"requested={requested_name} existing={data_conflict.name}"
        )

    before = {"object": armature.name, "data": armature.data.name}
    armature.data.name = requested_name
    armature.name = requested_name
    after = {"object": armature.name, "data": armature.data.name}
    if after != {"object": requested_name, "data": requested_name}:
        raise SystemExit(
            "Blender silently changed the normalized armature name: "
            f"requested={requested_name} actual={after}"
        )
    return before, after


def export_runtime(armature, mesh, output):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise SystemExit("Blender did not publish the conditioning GLB")


def main():
    args = parse_argv()
    source = require_input(args.input)
    reference = (
        require_input(args.reference_rig)
        if args.reference_rig is not None
        else None
    )
    output = require_output(args.output, "conditioning GLB")
    manifest = require_output(args.manifest, "conditioning manifest")
    if args.minimum_weight < 0.0:
        raise SystemExit("minimum weight must be non-negative")

    reference_authority = None
    if reference is not None:
        reference_authority = load_reference_deform_authority(
            reference, args.minimum_weight
        )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    armature, mesh = identify_runtime()
    temporary_weighted, weighted_vertices, maximum_influences = positive_weight_bones(
        mesh, armature, args.minimum_weight
    )
    authority_weighted = set(
        reference_authority["weighted_bones"] if reference_authority else []
    )
    target_bones = {bone.name for bone in armature.data.bones}
    missing_authority_bones = sorted(authority_weighted - target_bones)
    if missing_authority_bones:
        raise SystemExit(
            "conditioning skeleton is missing deform bones required by the "
            f"trusted reference rig: {missing_authority_bones}"
        )
    required_weighted = temporary_weighted | authority_weighted
    retained = include_ancestors(armature, required_weighted)
    roots_before = root_names(armature, {bone.name for bone in armature.data.bones})
    retained_roots_before = root_names(armature, retained)
    reparented_roots = []
    if len(retained_roots_before) != 1 and args.main_root:
        reparented_roots = parent_retained_roots_under_main(
            armature, retained, args.main_root
        )
    roots_after = root_names(armature, retained)
    if len(roots_after) != 1:
        raise SystemExit(
            "positive-weight skeleton plus ancestors must have exactly one root; "
            f"roots={roots_after}; pass an explicit trusted --main-root only "
            "when the additional roots are required weight-bearing components"
        )

    removed_bones = remove_unretained_bones(armature, retained)
    removed_groups = remove_unretained_vertex_groups(mesh, retained)
    removed_objects = remove_animation_and_extras(armature, mesh)
    armature_names_before, armature_names_after = normalize_armature_name(
        armature, args.armature_name
    )
    export_runtime(armature, mesh, output)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "mesh_authority": {
            "generated_topology_preserved": True,
            "mesh": mesh.name,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "weighted_vertices": weighted_vertices,
            "maximum_input_influences": maximum_influences,
            "material_slots": [
                slot.material.name if slot.material is not None else None
                for slot in mesh.material_slots
            ],
            "uv_layers": [layer.name for layer in mesh.data.uv_layers],
        },
        "skeleton_conditioning": {
            "method": "reference_deform_authority_plus_positive_weight_ancestors_v3",
            "reference_deform_authority": reference_authority,
            "roots_before": roots_before,
            "retained_roots_before_parenting": retained_roots_before,
            "root_after": roots_after[0],
            "explicit_main_root": args.main_root or None,
            "reparented_weight_bearing_roots": reparented_roots,
            "reparenting_preserved_armature_space_rest_coordinates": True,
            "temporary_transfer_weighted_bones": sorted(temporary_weighted),
            "reference_required_weighted_bones": sorted(authority_weighted),
            "weighted_bones": sorted(required_weighted),
            "retained_bones": sorted(retained),
            "removed_unretained_bones": removed_bones,
            "removed_vertex_groups": removed_groups,
            "removed_scene_objects": removed_objects,
            "animation_removed": True,
            "armature_name_normalization": {
                "reason": "skintokens_use_origin_export_name_compatibility",
                "before": armature_names_before,
                "after": armature_names_after,
            },
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "next_stage": "skintokens_use_skeleton_use_transfer",
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "FIXED_QUADRUPED_SKELETON_CONDITIONING_OK "
        f"root={roots_after[0]} retained={len(retained)} removed={len(removed_bones)} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
