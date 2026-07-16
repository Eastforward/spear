"""Combine two compatible rigged GLBs into one GLB with two named actions.

This is used for Flux/Hunyuan human meshes retargeted to Mixamo where each
source GLB currently carries one action. UE import is much more predictable if
the gate asset imports a single skeletal mesh with both loopable animations.
"""
from __future__ import annotations

import argparse
import os
import sys

import bpy


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--base-glb", required=True)
    p.add_argument("--append-glb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-action-name", default="Walking")
    p.add_argument("--append-action-name", default="Standing_Idle")
    return p.parse_args(argv)


def _objects_snapshot():
    return set(bpy.data.objects)


def _actions_snapshot():
    return set(bpy.data.actions)


def _import_glb(path):
    before_objects = _objects_snapshot()
    before_actions = _actions_snapshot()
    bpy.ops.import_scene.gltf(filepath=path)
    new_objects = [obj for obj in bpy.data.objects if obj not in before_objects]
    new_actions = [act for act in bpy.data.actions if act not in before_actions]
    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in new_objects if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes or len(new_actions) != 1:
        raise SystemExit(
            f"{path}: expected one armature, skinned mesh, and action; got "
            f"armatures={len(armatures)} meshes={len(meshes)} actions={len(new_actions)}"
        )
    armature = armatures[0]
    skinned_meshes = [
        mesh
        for mesh in meshes
        if mesh.parent is armature
        or any(
            modifier.type == "ARMATURE" and modifier.object is armature
            for modifier in mesh.modifiers
        )
    ]
    if len(skinned_meshes) != 1:
        raise SystemExit(
            f"{path}: expected one skinned runtime mesh; got {len(skinned_meshes)}"
        )
    return armature, skinned_meshes, new_actions[0], new_objects


def _stash_action_on_armature(armature, action, name):
    action.name = name
    action.use_fake_user = True
    armature.animation_data_create()
    track = armature.animation_data.nla_tracks.new()
    track.name = name
    start = int(round(action.frame_range[0]))
    strip = track.strips.new(name, start, action)
    strip.name = name


def _clear_imported_nla_tracks(armature):
    armature.animation_data_create()
    animation_data = armature.animation_data
    animation_data.action = None
    for track in list(animation_data.nla_tracks):
        animation_data.nla_tracks.remove(track)


def _remove_non_runtime_objects(objects, runtime_objects):
    runtime = set(runtime_objects)
    for obj in list(objects):
        if obj not in runtime:
            bpy.data.objects.remove(obj, do_unlink=True)


def _select_runtime_objects(armature, meshes):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature


def main():
    args = parse_argv()
    for path in (args.base_glb, args.append_glb):
        if not os.path.exists(path):
            raise SystemExit(f"missing input GLB: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    base_armature, base_meshes, base_action, base_objects = _import_glb(args.base_glb)
    append_armature, _append_meshes, append_action, append_objects = _import_glb(args.append_glb)

    _clear_imported_nla_tracks(base_armature)
    _stash_action_on_armature(base_armature, base_action, args.base_action_name)
    _stash_action_on_armature(base_armature, append_action, args.append_action_name)
    base_armature.animation_data.action = None

    _remove_non_runtime_objects(
        base_objects,
        (base_armature, *base_meshes),
    )
    _remove_non_runtime_objects(append_objects, ())
    _select_runtime_objects(base_armature, base_meshes)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_extra_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
    )
    print(
        f"COMBINE_GLB_ACTIONS_OK output={args.output} "
        f"actions={[args.base_action_name, args.append_action_name]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
