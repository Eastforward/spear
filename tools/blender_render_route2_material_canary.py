#!/usr/bin/env python3
"""Blender renderer for deterministic Route-2 material canary views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def review_view_directions() -> dict[str, tuple[float, float, float]]:
    return {
        "front": (0.0, -1.0, 0.0),
        "back": (0.0, 1.0, 0.0),
        "side": (1.0, 0.0, 0.0),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--public-relative-glb", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(argv)


def _blender_argv() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _remove_gltf_helpers(bpy: Any) -> None:
    collection = bpy.data.collections.get("glTF_not_exported")
    if collection is None:
        return
    for obj in list(collection.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def _setup_scene(bpy: Any, width: int, height: int):
    from mathutils import Vector

    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one rendered mesh, found {len(meshes)}")
    corners = [meshes[0].matrix_world @ Vector(corner) for corner in meshes[0].bound_box]
    minimum = Vector(tuple(min(point[index] for point in corners) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in corners) for index in range(3)))
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    radius = max(extent) * 2.2

    world = bpy.data.worlds.new("Route2MaterialWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.025,
        0.032,
        0.045,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    bpy.context.scene.world = world

    for name, direction, energy, size in (
        ("Key", (-0.7, -1.0, 1.2), 1100.0, 3.5),
        ("Fill", (0.9, -0.4, 0.5), 700.0, 3.0),
        ("Rim", (0.2, 1.0, 1.1), 900.0, 3.0),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        vector = Vector(direction).normalized()
        light.location = center + vector * radius
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    camera_data = bpy.data.cameras.new("Route2MaterialCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(float(extent.z) * 1.16, float(extent.x) * 1.45)
    camera_data.clip_start = 0.01
    camera_data.clip_end = radius * 5.0
    camera = bpy.data.objects.new("Route2MaterialCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    return center, radius, camera


def main(argv: list[str] | None = None) -> int:
    import bpy
    from mathutils import Vector

    args = parse_args(_blender_argv() if argv is None else argv)
    input_glb = args.input_glb.absolute()
    output_dir = args.output_dir.absolute()
    if input_glb.is_symlink() or not input_glb.is_file() or input_glb.resolve() != input_glb:
        raise RuntimeError("input GLB must be a direct regular file")
    if sha256_file(input_glb) != args.input_sha256:
        raise RuntimeError("input GLB hash changed before render")
    if output_dir.exists():
        raise RuntimeError("render output directory already exists")
    output_dir.mkdir(parents=True, mode=0o755)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.ops.import_scene.gltf(filepath=str(input_glb)) != {"FINISHED"}:
        raise RuntimeError("could not import deterministic material GLB")
    _remove_gltf_helpers(bpy)
    bpy.context.view_layer.update()
    center, radius, camera = _setup_scene(bpy, args.width, args.height)
    views: dict[str, dict[str, Any]] = {}
    for name, direction in review_view_directions().items():
        vector = Vector(direction)
        camera.location = center + vector * radius
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        path = output_dir / f"{name}.png"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite render: {path}")
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if not path.is_file() or path.stat().st_size <= 1024:
            raise RuntimeError(f"render output is missing or implausibly small: {path}")
        views[name] = {
            "filename": path.name,
            "direction": list(direction),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "resolution": [args.width, args.height],
        }
    manifest = {
        "schema": "route2_deterministic_material_render_v1",
        "asset_id": args.asset_id,
        "input_glb": {
            "relative_path": args.public_relative_glb,
            "sha256": args.input_sha256,
        },
        "canonical_front": "negative-y",
        "engine": "BLENDER_EEVEE_NEXT",
        "views": views,
    }
    _write_exclusive(
        output_dir / "render_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"ROUTE2_MATERIAL_RENDER_OK asset_id={args.asset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
