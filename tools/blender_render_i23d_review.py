#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys


def view_positions(center, radius, front_sign=-1):
    if front_sign not in {-1, 1}:
        raise ValueError("front_sign must be -1 or 1")
    x, y, z = center
    return {
        "front": (x, y + radius * front_sign, z),
        "side": (x + radius, y, z),
        "back": (x, y - radius * front_sign, z),
        "quarter": (
            x + radius * 0.72,
            y + radius * 0.72 * front_sign,
            z + radius * 0.08,
        ),
    }


def view_positions_for_axis(center, radius, front_axis):
    if front_axis == "negative-y":
        return view_positions(center, radius, front_sign=-1)
    if front_axis == "positive-y":
        return view_positions(center, radius, front_sign=1)
    if front_axis not in {"negative-x", "positive-x"}:
        raise ValueError(f"unsupported front axis: {front_axis}")
    x, y, z = center
    sign = 1 if front_axis == "positive-x" else -1
    return {
        "front": (x + radius * sign, y, z),
        "side": (x, y + radius, z),
        "back": (x - radius * sign, y, z),
        "quarter": (
            x + radius * 0.72 * sign,
            y + radius * 0.72,
            z + radius * 0.08,
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render static image-to-3D review views")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--include-top", action="store_true")
    parser.add_argument("--animal-material-preview", action="store_true")
    parser.add_argument(
        "--front-axis",
        choices=("negative-y", "positive-y", "negative-x", "positive-x"),
        default="negative-y",
    )
    return parser.parse_args(argv)


def _blender_argv():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def main(argv=None):
    import bpy
    from mathutils import Vector

    args = parse_args(_blender_argv() if argv is None else argv)
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if input_path.suffix.lower() not in {".glb", ".gltf"}:
        raise SystemExit(f"Unsupported input format: {input_path.suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise SystemExit(f"No mesh found in {input_path}")

    material_preview = {
        "mode": "raw_glb_material",
        "principled_nodes_changed": 0,
        "metallic_links_removed": 0,
        "roughness_links_removed": 0,
    }
    if args.animal_material_preview:
        material_preview["mode"] = "ue_animal_nonmetallic_roughness_preview_v1"
        materials = {
            slot.material
            for mesh in meshes
            for slot in mesh.material_slots
            if slot.material is not None
        }
        for material in materials:
            if not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                metallic = node.inputs.get("Metallic")
                roughness = node.inputs.get("Roughness")
                if metallic is None or roughness is None:
                    continue
                for link in list(metallic.links):
                    material.node_tree.links.remove(link)
                    material_preview["metallic_links_removed"] += 1
                for link in list(roughness.links):
                    material.node_tree.links.remove(link)
                    material_preview["roughness_links_removed"] += 1
                metallic.default_value = 0.0
                roughness.default_value = 0.95
                material_preview["principled_nodes_changed"] += 1

    corners = []
    for mesh in meshes:
        corners.extend(mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box)
    minimum = Vector(
        (
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )
    )
    center = (minimum + maximum) / 2
    extent = maximum - minimum
    radius = max(extent) * 1.7

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.06, 0.07, 0.09, 1.0)
    light_scale = 0.25 if args.animal_material_preview else 1.0
    background_strength = 0.25 if args.animal_material_preview else 0.5
    background.inputs["Strength"].default_value = background_strength

    light_specs = (
        ("Key", (center.x - radius, center.y - radius, center.z + radius), 1400, 4),
        ("Fill", (center.x + radius, center.y - radius * 0.2, center.z + radius * 0.4), 900, 3),
        ("Rim", (center.x, center.y + radius, center.z + radius), 1100, 3),
    )
    for name, location, energy, size in light_specs:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy * light_scale
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 58
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if args.animal_material_preview:
        scene.view_settings.exposure = -0.5

    views = view_positions_for_axis(tuple(center), radius, args.front_axis)
    if args.include_top:
        views["top"] = (center.x, center.y, center.z + radius)
    for name, location in views.items():
        camera.location = location
        camera_up = "X" if name == "top" else "Y"
        camera.rotation_euler = (center - camera.location).to_track_quat(
            "-Z", camera_up
        ).to_euler()
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

    manifest = {
        "input": str(input_path),
        "bbox_min": list(minimum),
        "bbox_max": list(maximum),
        "extent": list(extent),
        "front_axis": args.front_axis,
        "views": {name: list(location) for name, location in views.items()},
        "resolution": [args.width, args.height],
        "material_preview": material_preview,
        "lighting": {
            "area_light_scale": light_scale,
            "world_strength": background_strength,
            "exposure": float(scene.view_settings.exposure),
        },
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"I23D_STATIC_REVIEW_OK {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
