"""Render sampled frames from a rigged GLB animation in Blender.

Usage:
  blender --background --python tools/blender_render_glb_animation.py -- \\
    --input tmp/hy3d/swap_test/Dog_robust_swap.glb \\
    --action Walking \\
    --output-dir /tmp/anim_robust_swap \\
    --n-frames 12
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import mathutils


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--action", default="Walking")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--view", default="side", choices=["side", "front", "quarter"])
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   choices=["BLENDER_EEVEE_NEXT", "CYCLES"])
    return p.parse_args(argv)


def look_at(obj, target):
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def choose_action(action_name):
    actions = list(bpy.data.actions)
    for action in actions:
        if action.name == action_name:
            return action
    lowered = action_name.lower()
    for action in actions:
        if lowered in action.name.lower():
            return action
    raise SystemExit(f"missing action {action_name}; available={[a.name for a in actions]}")


def mesh_bbox(meshes):
    mn = [1e9, 1e9, 1e9]
    mx = [-1e9, -1e9, -1e9]
    for obj in meshes:
        for vertex in obj.data.vertices:
            p = obj.matrix_world @ vertex.co
            for axis in range(3):
                mn[axis] = min(mn[axis], p[axis])
                mx[axis] = max(mx[axis], p[axis])
    return mn, mx


def main():
    args = parse_argv()
    os.makedirs(args.output_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=args.input)
    else:
        raise SystemExit(f"unsupported input {ext}")

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not meshes or not armatures:
        raise SystemExit("input must contain a mesh and armature")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    for mesh in meshes:
        if mesh != body:
            mesh.hide_viewport = True
            mesh.hide_render = True
    armature = armatures[0]
    action = choose_action(args.action)
    armature.animation_data_create()
    armature.animation_data.action = action
    start, end = action.frame_range
    if end <= start:
        end = start + max(1, args.n_frames - 1)
    print(f"[anim] body={body.name} hidden_meshes={[m.name for m in meshes if m != body]} "
          f"armature={armature.name} "
          f"action={action.name} frame_range=({start:.2f}, {end:.2f})", flush=True)

    mn, mx = mesh_bbox([body])
    center = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    diag = math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3)))
    radius = diag * 2.0
    print(f"[anim] bbox_min={mn} bbox_max={mx} center={center} diag={diag:.3f}", flush=True)

    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (center[0], center[1] - radius, center[2] + radius)
    light.rotation_euler = (math.radians(60), 0, 0)

    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.60, 0.65, 1.0)
    bg.inputs[1].default_value = 0.7

    cam_data = bpy.data.cameras.new(name="Cam")
    cam_data.lens = 55
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    if args.view == "front":
        cam.location = (center[0] + radius, center[1], center[2] + diag * 0.12)
    elif args.view == "quarter":
        cam.location = (center[0] + radius * 0.85, center[1] - radius * 0.85, center[2] + diag * 0.18)
    else:
        cam.location = (center[0], center[1] - radius, center[2] + diag * 0.12)
    look_at(cam, center)

    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False
    if args.engine == "CYCLES":
        scene.cycles.samples = 32

    for i in range(args.n_frames):
        t = 0.0 if args.n_frames == 1 else i / (args.n_frames - 1)
        frame = start + (end - start) * t
        scene.frame_set(int(round(frame)))
        bpy.context.view_layer.update()
        scene.render.filepath = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[anim] frame {i + 1}/{args.n_frames} source_frame={frame:.2f}", flush=True)

    print(f"RENDER_GLB_ANIM_OK {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
