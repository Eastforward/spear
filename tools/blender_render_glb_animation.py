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


MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO = 0.35


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--action", default="Walking")
    p.add_argument(
        "--rest-pose",
        action="store_true",
        help="Render the authored armature rest pose instead of an animation action.",
    )
    p.add_argument(
        "--quadruped-far-limb-offset-ratio",
        type=float,
        default=0.0,
        help=(
            "For the Quaternius Cat/Dog authored rest pose, translate only the "
            "far-side front/hind limb chains by this fraction of the hip span "
            "so four grounded limbs remain visible from an exact side camera."
        ),
    )
    p.add_argument(
        "--pose-template-clay-color",
        default=None,
        help=(
            "Optional #RRGGBB uniform clay material for a rest-pose image "
            "template. This also enables smooth polygon shading so source "
            "material patches and low-poly facets cannot leak into FLUX.2."
        ),
    )
    p.add_argument(
        "--review-clay-color",
        default=None,
        help=(
            "Optional #RRGGBB material override for animation QA. This is "
            "render-only evidence for source GLBs whose vertex-color alpha "
            "material is not visible in headless Eevee; it never edits or "
            "exports the input asset."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--view", default="side", choices=["side", "front", "quarter"])
    p.add_argument(
        "--asset-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "Manual cardinal rotation of the complete rigged asset around Blender "
            "+Z. Only -90/0/90/180 are accepted; this is used to build reviewer "
            "direction candidates and never performs automatic inference."
        ),
    )
    p.add_argument(
        "--trajectory-distance-ratio",
        type=float,
        default=0.0,
        help=(
            "Translate the complete asset from left to right along world +X by "
            "this multiple of its rest-pose diagonal while sampling the action."
        ),
    )
    p.add_argument(
        "--orthographic",
        action="store_true",
        help="Use an orthographic camera for canonical pose-template evidence.",
    )
    p.add_argument(
        "--camera-distance-multiplier",
        type=float,
        default=2.0,
        help="Camera distance as a multiple of the rest-pose mesh diagonal.",
    )
    p.add_argument(
        "--ground-plane",
        action="store_true",
        help="Render a neutral floor at the rest-pose mesh minimum Z for foot-contact review.",
    )
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


def add_review_ground(center, ground_z, diag):
    bpy.ops.mesh.primitive_plane_add(
        size=max(2.0, diag * 4.0),
        location=(center[0], center[1], ground_z - diag * 0.002),
    )
    ground = bpy.context.object
    ground.name = "FootContactReviewGround"
    material = bpy.data.materials.new(name="FootContactReviewGroundMaterial")
    material.diffuse_color = (0.18, 0.20, 0.22, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.18, 0.20, 0.22, 1.0)
    principled.inputs["Roughness"].default_value = 0.85
    ground.data.materials.append(material)
    print(f"[ground] z={ground.location.z:.6f} size={max(2.0, diag * 4.0):.3f}", flush=True)
    return ground


def parse_hex_color(value):
    if (
        not isinstance(value, str)
        or len(value) != 7
        or not value.startswith("#")
    ):
        raise SystemExit("--pose-template-clay-color must use #RRGGBB")
    try:
        channels = tuple(int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as error:
        raise SystemExit("--pose-template-clay-color must use #RRGGBB") from error
    return (*channels, 1.0)


def apply_pose_template_clay_material(body, color):
    rgba = parse_hex_color(color)
    material = bpy.data.materials.new(name="PoseTemplateUniformClay")
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = 0.78
    body.data.materials.clear()
    body.data.materials.append(material)
    for polygon in body.data.polygons:
        polygon.use_smooth = True
    print(
        f"[pose-template] template_material=uniform_clay color={color.lower()} "
        "smooth_shading=true",
        flush=True,
    )


def apply_quadruped_far_limb_offset(armature, offset_ratio):
    """Expose far-side limbs without yawing the torso or the camera.

    Quaternius Cat.glb and Dog.glb share these authored bone names.  The
    positive-Y limbs are far from the renderer's side camera (located on -Y).
    Moving the far front chain slightly backward and the far hind chain
    slightly forward creates a compact neutral rigging stance rather than a
    walking stride.  The translations contain no Y/Z component: torso/head
    orientation and every paw's ground height remain unchanged.
    """
    names = ("Bone.017", "Bone.008")
    missing = sorted(set(names) - set(armature.pose.bones.keys()))
    if missing:
        raise SystemExit(f"quadruped far-limb bones are missing: {missing}")
    hip_span = abs(
        armature.pose.bones["Bone.017"].head.x
        - armature.pose.bones["Bone.008"].head.x
    )
    offset = hip_span * offset_ratio
    translations = {
        "Bone.017": -offset,
        "Bone.008": offset,
    }
    for name, delta_x in translations.items():
        pose_bone = armature.pose.bones[name]
        pose_bone.matrix = (
            mathutils.Matrix.Translation((delta_x, 0.0, 0.0)) @ pose_bone.matrix
        )
        bpy.context.view_layer.update()
    print(
        f"[pose-template] far_limb_offset_ratio={offset_ratio:.4f} "
        f"hip_span={hip_span:.6f} offset={offset:.6f} "
        "torso_transform=identity camera_yaw_deg=0 ground_delta=0",
        flush=True,
    )


def apply_asset_cardinal_yaw(armature, yaw_deg):
    """Rotate the whole skinned asset without changing any bone animation."""
    pivot = armature.matrix_world.translation.copy()
    asset_root = bpy.data.objects.new("ManualCardinalAssetRoot", None)
    bpy.context.collection.objects.link(asset_root)
    asset_root.location = pivot
    original_world = armature.matrix_world.copy()
    armature.parent = asset_root
    armature.matrix_world = original_world
    asset_root.rotation_euler.z = math.radians(float(yaw_deg))
    bpy.context.view_layer.update()
    print(
        f"[direction-candidate] asset_yaw_deg={float(yaw_deg):+.0f} "
        "axis=blender_positive_z automatic_direction_inference=false",
        flush=True,
    )
    return asset_root


def main():
    args = parse_argv()
    if not 0.75 <= args.camera_distance_multiplier <= 4.0:
        raise SystemExit("--camera-distance-multiplier must be in [0.75, 4.0]")
    if not 0.0 <= args.quadruped_far_limb_offset_ratio <= (
        MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO
    ):
        raise SystemExit(
            "--quadruped-far-limb-offset-ratio must be in "
            f"[0, {MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO}]"
        )
    if args.quadruped_far_limb_offset_ratio and not args.rest_pose:
        raise SystemExit("--quadruped-far-limb-offset-ratio requires --rest-pose")
    if args.pose_template_clay_color and not args.rest_pose:
        raise SystemExit("--pose-template-clay-color requires --rest-pose")
    if args.pose_template_clay_color and args.review_clay_color:
        raise SystemExit("choose only one clay material override")
    allowed_yaws = (-90.0, 0.0, 90.0, 180.0)
    if not any(math.isclose(args.asset_yaw_deg, value, abs_tol=1e-9) for value in allowed_yaws):
        raise SystemExit("asset_yaw_deg must be one of -90, 0, 90, 180")
    if not 0.0 <= args.trajectory_distance_ratio <= 2.0:
        raise SystemExit("--trajectory-distance-ratio must be in [0, 2]")
    if args.rest_pose and args.trajectory_distance_ratio:
        raise SystemExit("--trajectory-distance-ratio is only valid for animation actions")
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
    asset_root = apply_asset_cardinal_yaw(armature, args.asset_yaw_deg)
    if args.pose_template_clay_color:
        apply_pose_template_clay_material(body, args.pose_template_clay_color)
    if args.review_clay_color:
        apply_pose_template_clay_material(body, args.review_clay_color)
    if args.rest_pose:
        armature.animation_data_clear()
        if args.quadruped_far_limb_offset_ratio:
            armature.data.pose_position = "POSE"
            for pose_bone in armature.pose.bones:
                pose_bone.matrix_basis.identity()
            bpy.context.view_layer.update()
            apply_quadruped_far_limb_offset(
                armature, args.quadruped_far_limb_offset_ratio
            )
        else:
            armature.data.pose_position = "REST"
        action = None
        action_label = (
            "authored_rest_pose_with_far_limb_offset"
            if args.quadruped_far_limb_offset_ratio
            else "authored_rest_pose"
        )
        start = end = 1.0
    else:
        action = choose_action(args.action)
        armature.animation_data_create()
        armature.animation_data.action = action
        action_label = action.name
        start, end = action.frame_range
        if end <= start:
            end = start + max(1, args.n_frames - 1)
    bpy.context.view_layer.update()
    print(f"[anim] body={body.name} hidden_meshes={[m.name for m in meshes if m != body]} "
          f"armature={armature.name} "
          f"action={action_label} frame_range=({start:.2f}, {end:.2f})", flush=True)

    mn, mx = mesh_bbox([body])
    center = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    diag = math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3)))
    radius = diag * args.camera_distance_multiplier
    radius += diag * 0.5 * args.trajectory_distance_ratio
    print(f"[anim] bbox_min={mn} bbox_max={mx} center={center} diag={diag:.3f}", flush=True)
    if args.ground_plane:
        add_review_ground(center, mn[2], diag)

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
    if args.orthographic:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = diag * 1.05
    else:
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

    trajectory_base_location = asset_root.location.copy()
    trajectory_distance = diag * args.trajectory_distance_ratio
    if trajectory_distance:
        print(
            f"[direction-candidate] trajectory_axis=world_positive_x "
            f"distance={trajectory_distance:.6f} start_fraction=-0.5 end_fraction=0.5",
            flush=True,
        )
    for i in range(args.n_frames):
        t = 0.0 if args.n_frames == 1 else i / (args.n_frames - 1)
        frame = start + (end - start) * t
        scene.frame_set(int(round(frame)))
        asset_root.location = trajectory_base_location + mathutils.Vector(
            ((t - 0.5) * trajectory_distance, 0.0, 0.0)
        )
        bpy.context.view_layer.update()
        scene.render.filepath = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[anim] frame {i + 1}/{args.n_frames} source_frame={frame:.2f}", flush=True)

    print(f"RENDER_GLB_ANIM_OK {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
