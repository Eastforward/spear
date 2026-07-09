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
    if not armatures or not meshes or not new_actions:
        raise SystemExit(
            f"{path}: expected armature, mesh, and action; got "
            f"armatures={len(armatures)} meshes={len(meshes)} actions={len(new_actions)}"
        )
    return armatures[0], meshes, new_actions[0], new_objects


def _stash_action_on_armature(armature, action, name):
    action.name = name
    action.use_fake_user = True
    armature.animation_data_create()
    track = armature.animation_data.nla_tracks.new()
    track.name = name
    start = int(round(action.frame_range[0]))
    strip = track.strips.new(name, start, action)
    strip.name = name


def main():
    args = parse_argv()
    for path in (args.base_glb, args.append_glb):
        if not os.path.exists(path):
            raise SystemExit(f"missing input GLB: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    base_armature, _base_meshes, base_action, _base_objects = _import_glb(args.base_glb)
    append_armature, _append_meshes, append_action, append_objects = _import_glb(args.append_glb)

    _stash_action_on_armature(base_armature, base_action, args.base_action_name)
    _stash_action_on_armature(base_armature, append_action, args.append_action_name)
    base_armature.animation_data.action = base_action

    for obj in append_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_extra_animations=True,
    )
    print(
        f"COMBINE_GLB_ACTIONS_OK output={args.output} "
        f"actions={[args.base_action_name, args.append_action_name]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
