"""Copy Walk/Idle actions between compatible semantic animal skeletons.

This is for stable Quaternius templates that share exact pose-bone names but
ship different action subsets.  It never transfers generated mesh weights and
never replaces target geometry: the target template's topology, armature,
weights, and materials remain authoritative.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import bpy


SCHEMA = "avengine_compatible_animal_action_transfer_v1"
BONE_PATH = re.compile(r'pose\.bones\["([^"]+)"\]')
OPTIONAL_MISSING_BONE_PREFIXES = ("Tail",)


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-glb", required=True)
    parser.add_argument("--source-glb", required=True)
    parser.add_argument("--source-walk-action", default="Walk")
    parser.add_argument("--target-idle-action", default="Idle")
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_glb(path: Path):
    old_objects = set(bpy.data.objects)
    old_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [item for item in bpy.data.objects if item not in old_objects]
    actions = [item for item in bpy.data.actions if item not in old_actions]
    armatures = [item for item in objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit(f"{path}: expected exactly one armature")
    armature = armatures[0]
    skinned_meshes = [
        item
        for item in objects
        if item.type == "MESH"
        and (
            item.parent is armature
            or any(
                modifier.type == "ARMATURE" and modifier.object is armature
                for modifier in item.modifiers
            )
        )
    ]
    if not skinned_meshes:
        raise SystemExit(f"{path}: no skinned mesh")
    return armature, skinned_meshes, actions, objects


def choose_action(actions, requested):
    for action in actions:
        if action.name == requested:
            return action
    matches = [action for action in actions if requested.lower() in action.name.lower()]
    if len(matches) != 1:
        raise SystemExit(
            f"action {requested!r} did not resolve uniquely; "
            f"available={[action.name for action in actions]}"
        )
    return matches[0]


def action_bones(action):
    names = set()
    for curve in action.fcurves:
        match = BONE_PATH.search(curve.data_path)
        if match:
            names.add(match.group(1))
    return names


def drop_optional_missing_bone_curves(action, missing_bones):
    dropped = []
    for curve in list(action.fcurves):
        match = BONE_PATH.search(curve.data_path)
        if match and match.group(1) in missing_bones:
            dropped.append(curve.data_path)
            action.fcurves.remove(curve)
    return dropped


def clear_nla(armature):
    armature.animation_data_create()
    armature.animation_data.action = None
    for track in list(armature.animation_data.nla_tracks):
        armature.animation_data.nla_tracks.remove(track)


def add_action_track(armature, action, name):
    action.name = name
    action.use_fake_user = True
    track = armature.animation_data.nla_tracks.new()
    track.name = name
    strip = track.strips.new(name, int(round(action.frame_range[0])), action)
    strip.name = name


def remove_objects(objects):
    for item in list(objects):
        bpy.data.objects.remove(item, do_unlink=True)


def main():
    args = parse_argv()
    target_path = Path(args.target_glb).resolve()
    source_path = Path(args.source_glb).resolve()
    output_glb = Path(args.output_glb).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    for path in (target_path, source_path):
        if path.suffix.lower() != ".glb" or not path.is_file():
            raise SystemExit(f"missing or unsupported input: {path}")
    for path in (output_glb, output_manifest):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to replace output: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    target_armature, target_meshes, target_actions, target_objects = import_glb(
        target_path
    )
    source_armature, _source_meshes, source_actions, source_objects = import_glb(
        source_path
    )
    target_idle = choose_action(target_actions, args.target_idle_action)
    source_walk = choose_action(source_actions, args.source_walk_action)

    target_bones = {bone.name for bone in target_armature.data.bones}
    source_bones = {bone.name for bone in source_armature.data.bones}
    used_walk_bones = action_bones(source_walk)
    missing_target_bones = sorted(used_walk_bones - target_bones)
    required_missing_bones = [
        name
        for name in missing_target_bones
        if not name.startswith(OPTIONAL_MISSING_BONE_PREFIXES)
    ]
    if required_missing_bones:
        raise SystemExit(
            "source walk animates required bones missing from target: "
            f"{required_missing_bones}"
        )

    transferred_walk = source_walk.copy()
    dropped_optional_curves = drop_optional_missing_bone_curves(
        transferred_walk, set(missing_target_bones)
    )
    clear_nla(target_armature)
    add_action_track(target_armature, target_idle, "Idle")
    add_action_track(target_armature, transferred_walk, "Walking")
    target_armature.animation_data.action = None
    remove_objects(source_objects)

    bpy.ops.object.select_all(action="DESELECT")
    target_armature.select_set(True)
    for mesh in target_meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_extra_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "geometry_and_weights_authority": True,
        },
        "action_source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "resolved_walk_action": source_walk.name,
            "geometry_used": False,
            "weights_used": False,
        },
        "skeleton_compatibility": {
            "target_bone_count": len(target_bones),
            "source_bone_count": len(source_bones),
            "common_bone_count": len(target_bones & source_bones),
            "walk_animated_bones": sorted(used_walk_bones),
            "missing_target_bones": missing_target_bones,
            "required_missing_bones": required_missing_bones,
            "optional_missing_bone_prefixes": list(OPTIONAL_MISSING_BONE_PREFIXES),
            "dropped_optional_bone_curve_count": len(dropped_optional_curves),
            "dropped_optional_bone_curves": dropped_optional_curves,
        },
        "output": {
            "path": str(output_glb),
            "sha256": sha256_file(output_glb),
            "actions": ["Idle", "Walking"],
        },
        "status": "research_candidate_pending_deformation_and_media_qa",
        "formal_dataset_registration_authorized": False,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        f"COMPATIBLE_ANIMAL_ACTION_TRANSFER_OK output={output_glb} "
        f"common_bones={len(target_bones & source_bones)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
