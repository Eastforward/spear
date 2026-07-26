"""Render a deterministic turntable of a rigged animal GLB for forward review.

Imports the GLB once and renders N evenly spaced camera azimuths around the
world +Z axis at a fixed elevation, plus one head-candidate hero shot per
requested candidate yaw (camera placed on that azimuth looking back at the
mesh center, so the face is visible exactly when that end is the head).
Frames feed the local forward-review page; nothing here decides anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import bpy
import numpy as np


SCHEMA = "avengine_forward_turntable_render_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument(
        "--candidate-yaw-deg", type=float, action="append", default=[]
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_bounds():
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                points.append(obj.matrix_world @ obj.matrix_world.to_3x3().__class__().col[0].__class__(corner))
    return points


def mesh_world_bounds():
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        for corner in obj.bound_box:
            world = matrix[:3, :3] @ np.asarray(corner) + matrix[:3, 3]
            minimum = np.minimum(minimum, world)
            maximum = np.maximum(maximum, world)
    if not np.all(np.isfinite(minimum)):
        raise SystemExit("scene has no mesh bounds")
    return minimum, maximum


def place_camera(camera, center, radius, azimuth_deg, elevation_deg=18.0):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    position = center + radius*np.asarray(
        [
            math.cos(elevation)*math.cos(azimuth),
            math.cos(elevation)*math.sin(azimuth),
            math.sin(elevation),
        ]
    )
    camera.location = position.tolist()
    direction = center - position
    import mathutils

    camera.rotation_euler = (
        mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()
    )


def main():
    args = parse_argv()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input GLB: {source}")
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"refusing to replace output dir: {output_dir}")
    if not 8 <= args.frames <= 120:
        raise SystemExit("--frames must be in [8, 120]")
    output_dir.mkdir(parents=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    minimum, maximum = mesh_world_bounds()
    center = (minimum + maximum)/2.0
    radius = float(np.linalg.norm(maximum - minimum))*1.1

    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("turntable_camera")
    camera = bpy.data.objects.new("turntable_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    light_data = bpy.data.lights.new("turntable_key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("turntable_key", light_data)
    light.rotation_euler = (math.radians(-35.0), math.radians(20.0), 0.0)
    scene.collection.objects.link(light)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("turntable_world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.12, 0.12, 0.14, 1.0)

    frames = []
    for index in range(args.frames):
        azimuth = 360.0*index/args.frames
        place_camera(camera, center, radius, azimuth)
        frame_path = output_dir / f"turntable_{index:03d}.png"
        scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        frames.append(
            {"index": index, "azimuth_deg": azimuth, "path": frame_path.name}
        )

    candidates = []
    for yaw in args.candidate_yaw_deg:
        place_camera(camera, center, radius*0.85, yaw, elevation_deg=8.0)
        hero_path = output_dir / f"candidate_{int(round(yaw)) % 360:03d}.png"
        scene.render.filepath = str(hero_path)
        bpy.ops.render.render(write_still=True)
        candidates.append({"candidate_yaw_deg": yaw, "path": hero_path.name})

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "camera": {
            "orbit_axis": "world_positive_z",
            "elevation_deg": 18.0,
            "candidate_elevation_deg": 8.0,
        },
        "frames": frames,
        "candidates": candidates,
        "resolution": [args.width, args.height],
    }
    manifest_path = output_dir / "turntable_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "FORWARD_TURNTABLE_OK "
        f"frames={len(frames)} candidates={len(candidates)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
