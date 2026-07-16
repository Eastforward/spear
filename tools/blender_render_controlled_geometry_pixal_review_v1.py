#!/usr/bin/env python3
"""Render immutable Front/Back/Side/Top/Quarter evidence for a Pixal GLB."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


SCHEMA = "route2_controlled_geometry_pixal_static_review_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
PIXAL_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
WIDTH = 600
HEIGHT = 800
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class ReviewError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, public_path: Path | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        raise ReviewError(f"artifact must be a direct nonempty file: {path}")
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, description: str) -> dict[str, Any]:
    path = Path(path).absolute()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"{description} is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ReviewError(f"{description} must contain an object")
    return payload


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise ReviewError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def view_positions(center: Sequence[float], radius: float) -> dict[str, tuple[float, float, float]]:
    x, y, z = (float(value) for value in center)
    return {
        "front": (x, y + radius, z),
        "back": (x, y - radius, z),
        "side": (x + radius, y, z),
        "quarter": (x + radius * 0.72, y + radius * 0.72, z + radius * 0.08),
        "top": (x, y, z + radius),
    }


def _panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").resize((300, 400), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bounds[2] + 18, bounds[3] + 18), fill=(0, 0, 0))
    draw.text((13, 13), label, font=font, fill=(255, 255, 255))
    return panel


def make_contact_sheet(reference: Image.Image, views: Mapping[str, Image.Image]) -> Image.Image:
    panels = [("approved 2D", reference), *[(name, views[name]) for name in ("front", "back", "side", "top", "quarter")]]
    canvas = Image.new("RGB", (900, 800), (28, 28, 28))
    for index, (label, image) in enumerate(panels):
        canvas.paste(_panel(image, label), ((index % 3) * 300, (index // 3) * 400))
    return canvas


def authenticate_inputs(asset_id: str, glb: Path, manifest_path: Path) -> dict[str, Any]:
    asset_root = PIXAL_ROOT / asset_id
    glb = Path(glb).absolute()
    manifest_path = Path(manifest_path).absolute()
    if (
        asset_root.name != asset_id
        or glb != asset_root / "canary_1024_seed42.glb"
        or manifest_path != asset_root / "canary_1024_seed42.manifest.json"
    ):
        raise ReviewError("Pixal review inputs are not canonical")
    manifest = load_json(manifest_path, "controlled Pixal manifest")
    if (
        manifest.get("schema") != "route2_controlled_geometry_pixal_candidate_v1"
        or manifest.get("asset_id") != asset_id
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("parameters")
        != {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
        or manifest.get("output") != record(glb)
        or manifest.get("pbr_glb_readback", {}).get("passed") is not True
    ):
        raise ReviewError("controlled Pixal manifest identity/PBR state changed")
    reference = Path(manifest["input_rgba"]["path"])
    if manifest["input_rgba"] != record(reference):
        raise ReviewError("controlled Pixal reference RGBA changed")
    attempt_path = asset_root / "pixal_attempt.json"
    attempt = load_json(attempt_path, "controlled Pixal attempt")
    if (
        attempt.get("schema") != "route2_controlled_geometry_pixal_attempt_v1"
        or attempt.get("asset_id") != asset_id
        or attempt.get("status") != "succeeded_pending_static_multiview_qa"
        or attempt.get("static_multiview_qa") != "pending"
        or attempt.get("glb") != record(glb)
        or attempt.get("manifest") != record(manifest_path)
    ):
        raise ReviewError("controlled Pixal attempt lineage changed")
    return {
        "manifest": manifest,
        "manifest_record": record(manifest_path),
        "glb_record": record(glb),
        "attempt_record": record(attempt_path),
        "reference": reference,
        "reference_record": record(reference),
    }


def render(asset_id: str, glb: Path, manifest_path: Path, output_dir: Path) -> Path:
    import bpy
    from mathutils import Vector

    authenticated = authenticate_inputs(asset_id, glb, manifest_path)
    output_dir = Path(output_dir).absolute()
    asset_root = PIXAL_ROOT / asset_id
    if output_dir != asset_root / "static_review_v1" or os.path.lexists(output_dir):
        raise ReviewError("static review output must be the unused canonical directory")
    staging = Path(tempfile.mkdtemp(prefix=".static_review_v1.", suffix=".staging", dir=asset_root))
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        result = bpy.ops.import_scene.gltf(filepath=str(Path(glb).absolute()))
        if "FINISHED" not in result:
            raise ReviewError("Blender could not import controlled Pixal GLB")
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not meshes:
            raise ReviewError("controlled Pixal GLB imported no mesh")
        corners = [mesh.matrix_world @ Vector(corner) for mesh in meshes for corner in mesh.bound_box]
        minimum = Vector(tuple(min(point[index] for point in corners) for index in range(3)))
        maximum = Vector(tuple(max(point[index] for point in corners) for index in range(3)))
        extent = maximum - minimum
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in extent):
            raise ReviewError("controlled Pixal GLB has invalid extents")
        center = (minimum + maximum) / 2
        radius = max(extent) * 1.7
        world = bpy.data.worlds.new("ControlledPixalReviewWorld")
        bpy.context.scene.world = world
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        background.inputs["Color"].default_value = (0.055, 0.065, 0.085, 1.0)
        background.inputs["Strength"].default_value = 0.55
        for name, location, energy, size in (
            ("Key", (center.x - radius, center.y + radius, center.z + radius), 1400, 4),
            ("Fill", (center.x + radius, center.y + radius * 0.2, center.z + radius * 0.4), 900, 3),
            ("Rim", (center.x, center.y - radius, center.z + radius), 1100, 3),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy = energy
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
        scene.render.resolution_x = WIDTH
        scene.render.resolution_y = HEIGHT
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        views = view_positions(center, radius)
        for name, location in views.items():
            camera.location = location
            up = "Y" if name != "top" else "X"
            camera.rotation_euler = (center - camera.location).to_track_quat("-Z", up).to_euler()
            scene.render.filepath = str(staging / f"{name}.png")
            bpy.ops.render.render(write_still=True)
        loaded_views = {}
        for name in views:
            with Image.open(staging / f"{name}.png") as opened:
                opened.load()
                loaded_views[name] = opened.convert("RGB")
        with Image.open(authenticated["reference"]) as opened:
            opened.load()
            reference = opened.convert("RGBA")
            backdrop = Image.new("RGB", reference.size, (210, 210, 210))
            backdrop.paste(reference.convert("RGB"), mask=reference.getchannel("A"))
        contact = make_contact_sheet(backdrop, loaded_views)
        contact.save(staging / "contact_sheet.png", format="PNG")
        public = output_dir
        artifacts = {
            filename: record(staging / filename, public_path=public / filename)
            for filename in sorted([*(f"{name}.png" for name in views), "contact_sheet.png"])
        }
        materials = [slot.material for mesh in meshes for slot in mesh.material_slots if slot.material]
        payload = {
            "schema": SCHEMA,
            "asset_id": asset_id,
            "state_classification": "research_candidate",
            "formal_registration_authorized": False,
            "front_axis": "positive-y",
            "up_axis": "positive-z",
            "input": {
                key: value for key, value in authenticated.items() if key != "reference"
            },
            "runner": record(RUNNER_PATH),
            "geometry": {
                "mesh_object_count": len(meshes),
                "vertex_count": sum(len(mesh.data.vertices) for mesh in meshes),
                "polygon_count": sum(len(mesh.data.polygons) for mesh in meshes),
                "material_slot_count": sum(len(mesh.material_slots) for mesh in meshes),
                "material_count": len({material.name for material in materials}),
                "image_count": len(bpy.data.images),
                "bbox_min": list(minimum),
                "bbox_max": list(maximum),
                "extent": list(extent),
            },
            "views": {name: list(location) for name, location in views.items()},
            "resolution": [WIDTH, HEIGHT],
            "automatic_checks": {
                "glb_imported": True,
                "nonempty_mesh": True,
                "finite_positive_extents": True,
                "materials_present": bool(materials),
                "images_present": len(bpy.data.images) > 0,
            },
            "agent_visual_qa": "pending",
            "artifacts": artifacts,
        }
        (staging / "review_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in staging.iterdir():
            path.chmod(0o444)
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _rename_noreplace(staging, output_dir)
        print(f"CONTROLLED_PIXAL_STATIC_REVIEW_OK {output_dir}")
        return output_dir
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--pixal-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


if __name__ == "__main__":
    args = parse_args(_blender_argv())
    render(args.asset_id, args.input_glb, args.pixal_manifest, args.output_dir)
