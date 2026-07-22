#!/usr/bin/env python3
"""Render fixed rest-pose coat-edit views from one generated animal GLB.

The renderer never edits or exports the source asset.  It produces four
orthographic, same-lighting views whose exact camera contract can be recreated
by the multiview coat projector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


SCHEMA = "avengine_generated_animal_coat_views_v1"
VIEW_ORDER = ("front", "back", "left", "right")


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--front-axis",
        choices=("negative-x", "positive-x", "negative-y", "positive-y"),
        required=True,
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv)


def require_new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace output directory: {path}")
    path.mkdir(parents=True)
    return path


def primary_skinned_mesh():
    candidates = [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH"
        and item.vertex_groups
        and any(modifier.type == "ARMATURE" for modifier in item.modifiers)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one generated skinned mesh, got {len(candidates)}")
    return candidates[0]


def cardinal_vectors(front_axis: str):
    front = {
        "negative-x": Vector((-1.0, 0.0, 0.0)),
        "positive-x": Vector((1.0, 0.0, 0.0)),
        "negative-y": Vector((0.0, -1.0, 0.0)),
        "positive-y": Vector((0.0, 1.0, 0.0)),
    }[front_axis]
    up = Vector((0.0, 0.0, 1.0))
    left = up.cross(front).normalized()
    return front, -front, left, -left


def mesh_world_bounds(mesh):
    points = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def make_area_light(name, location, target, energy, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()
    return light


def main():
    args = parse_argv()
    if args.width != args.height or not 256 <= args.width <= 2048:
        raise RuntimeError("coat views require a square resolution in [256, 2048]")
    if not 1 <= args.samples <= 256:
        raise RuntimeError("--samples must be in [1, 256]")
    input_glb = args.input_glb.resolve()
    if input_glb.suffix.lower() != ".glb" or not input_glb.is_file():
        raise RuntimeError(f"missing input GLB: {input_glb}")
    output_dir = require_new_directory(args.output_dir)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    mesh = primary_skinned_mesh()
    for armature in [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]:
        armature.data.pose_position = "REST"
    for other in [item for item in bpy.context.scene.objects if item.type == "MESH"]:
        other.hide_render = other != mesh
    preview_material_changes = 0
    for material in {
        slot.material for slot in mesh.material_slots if slot.material is not None
    }:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            metallic = node.inputs.get("Metallic")
            roughness = node.inputs.get("Roughness")
            if metallic is None or roughness is None:
                continue
            for socket in (metallic, roughness):
                for link in list(socket.links):
                    material.node_tree.links.remove(link)
            metallic.default_value = 0.0
            roughness.default_value = 0.82
            preview_material_changes += 1
    bpy.context.view_layer.update()

    minimum, maximum = mesh_world_bounds(mesh)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    largest_extent = max(extent)
    distance = largest_extent * 3.0
    ortho_scale = largest_extent * 1.18

    world = bpy.data.worlds.new("NeutralCoatReviewWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
    background.inputs["Strength"].default_value = 0.30
    bpy.context.scene.world = world

    light_size = largest_extent * 2.0
    make_area_light(
        "CoatKey",
        center + Vector((-1.3, -1.1, 1.6)) * largest_extent,
        center,
        180.0,
        light_size,
    )
    make_area_light(
        "CoatFill",
        center + Vector((1.2, 1.0, 0.8)) * largest_extent,
        center,
        90.0,
        light_size,
    )

    camera_data = bpy.data.cameras.new("CoatProjectionCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera = bpy.data.objects.new("CoatProjectionCamera", camera_data)
    bpy.context.collection.objects.link(camera)

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = args.samples
    scene.view_settings.look = "AgX - Base Contrast"
    scene.view_settings.exposure = -0.7

    vectors = cardinal_vectors(args.front_axis)
    views = {}
    for name, direction in zip(VIEW_ORDER, vectors):
        camera.location = center + direction * distance
        camera.rotation_euler = (center - camera.location).to_track_quat(
            "-Z", "Y"
        ).to_euler()
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        views[name] = {
            "camera_location": list(camera.location),
            "camera_direction_to_subject": list((center - camera.location).normalized()),
        }

    manifest = {
        "schema": SCHEMA,
        "input_glb": str(input_glb),
        "front_axis": args.front_axis,
        "view_order": list(VIEW_ORDER),
        "resolution": [args.width, args.height],
        "samples": args.samples,
        "rest_pose": True,
        "primary_mesh": mesh.name,
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "bbox_min": list(minimum),
        "bbox_max": list(maximum),
        "center": list(center),
        "largest_extent": float(largest_extent),
        "camera_type": "ORTHO",
        "camera_distance": float(distance),
        "ortho_scale": float(ortho_scale),
        "views": views,
        "lighting": "fixed_neutral_two_area_v1",
        "render_only_material_preview": {
            "principled_nodes_changed": preview_material_changes,
            "metallic": 0.0,
            "roughness": 0.82,
            "exposure": -0.7,
        },
        "source_asset_modified": False,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"GENERATED_ANIMAL_COAT_VIEWS_OK output={output_dir}")


if __name__ == "__main__":
    main()
