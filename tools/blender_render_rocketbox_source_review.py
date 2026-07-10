"""Render an untouched Rocketbox avatar for the first human review gate.

Run with Blender:

    blender --background --python tools/blender_render_rocketbox_source_review.py -- \
      --asset-id rocketbox_male_adult_01 --fbx /absolute/avatar.fbx \
      --texture-dir /absolute/Textures --output-dir /absolute/review
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


VIEWS = {
    "front": (0.0, -1.0, 0.15),
    "back": (0.0, 1.0, 0.15),
    "left": (-1.0, 0.0, 0.15),
    "right": (1.0, 0.0, 0.15),
    "top": (0.0, 0.0, 1.0),
}
BODY_VIEW_SIZE = (1200, 1600)
CLOSE_VIEW_SIZE = (1200, 900)
VIDEO_SIZE = (1280, 720)
BODY_VIEW_FILES = {
    "front": "front.png",
    "back": "back.png",
    "left": "left.png",
    "right": "right.png",
    "top": "top.png",
}
CLOSE_VIEW_FILES = {
    "face_close": "face_close.png",
    "arms_close": "arms_close.png",
    "feet_close": "feet_close.png",
}
VIDEO_FILE = "turntable.mp4"
MANIFEST_FILE = "render_manifest.json"
LEGEND_CONTRACT = ("UP +Z", "FRONT -Y", "REST POSE / NO ACTION")
REVIEW_LIGHT_ENERGIES = (320.0, 140.0, 220.0)
BODY_FRAME_MARGIN = 1.24
LEGEND_FONT_SCALE = 0.018
LEGEND_MARGIN_SCALE = 0.070
OPACITY_ROUGHNESS = 0.86
ASSET_PREFIXES = {
    "rocketbox_male_adult_01": "m002",
    "rocketbox_female_adult_01": "f001",
}


@dataclass(frozen=True)
class ImportedAvatar:
    mesh: bpy.types.Object
    armature: bpy.types.Object
    imported_objects: tuple[bpy.types.Object, ...]
    material_slot_names: tuple[str, ...]


@dataclass(frozen=True)
class Bounds:
    minimum: Vector
    maximum: Vector
    corners: tuple[Vector, ...]

    @property
    def center(self) -> Vector:
        return (self.minimum + self.maximum) * 0.5

    @property
    def dimensions(self) -> Vector:
        return self.maximum - self.minimum


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", choices=tuple(ASSET_PREFIXES), required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forward-axis", default="-Y")
    parser.add_argument("--up-axis", default="+Z")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_avatar(fbx_path: Path, texture_prefix: str) -> ImportedAvatar:
    if not fbx_path.is_file():
        raise FileNotFoundError(f"Rocketbox FBX does not exist: {fbx_path}")
    before_objects = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    imported_objects = tuple(obj for obj in bpy.data.objects if obj not in before_objects)
    armatures = [obj for obj in imported_objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    if not armatures or not meshes:
        raise RuntimeError(f"Rocketbox FBX did not contain an armature and mesh: {fbx_path}")

    armature = armatures[0]
    mesh = max(meshes, key=lambda obj: len(obj.data.vertices))
    for obj in imported_objects:
        obj.animation_data_clear()
    for action in tuple(bpy.data.actions):
        bpy.data.actions.remove(action)
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    if len(armature.data.bones) != 80:
        raise RuntimeError(
            f"expected 80 Rocketbox avatar bones, got {len(armature.data.bones)}"
        )
    expected_material_names = [
        f"{texture_prefix}_body",
        f"{texture_prefix}_head",
        f"{texture_prefix}_opacity",
    ]
    material_slot_names = [slot.material.name for slot in mesh.material_slots]
    if material_slot_names != expected_material_names:
        raise RuntimeError(
            "unexpected Rocketbox material slots: "
            f"actual={material_slot_names} expected={expected_material_names}"
        )
    if not mesh.data.uv_layers:
        raise RuntimeError(f"Rocketbox mesh has no UV layer: {mesh.name}")
    return ImportedAvatar(
        mesh=mesh,
        armature=armature,
        imported_objects=imported_objects,
        material_slot_names=tuple(material_slot_names),
    )


def require_texture(texture_dir: Path, filename: str) -> Path:
    path = texture_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"required Rocketbox texture does not exist: {path}")
    return path


def load_image(path: Path, color_space: str) -> bpy.types.Image:
    image = bpy.data.images.load(str(path), check_existing=False)
    if color_space == "sRGB":
        image.colorspace_settings.name = "sRGB"
    else:
        image.colorspace_settings.name = "Non-Color"
    return image


def new_image_node(
    nodes: bpy.types.Nodes, path: Path, color_space: str, label: str
) -> bpy.types.Node:
    node = nodes.new("ShaderNodeTexImage")
    node.name = label
    node.label = label
    node.image = load_image(path, color_space)
    node.interpolation = "Linear"
    return node


def image_has_useful_alpha(image: bpy.types.Image) -> bool:
    if image.channels < 4:
        return False
    pixel_values = array("f", [0.0]) * len(image.pixels)
    image.pixels.foreach_get(pixel_values)
    pixel_count = len(pixel_values) // image.channels
    if pixel_count == 0:
        return False
    step = max(1, pixel_count // 4096)
    minimum = 1.0
    maximum = 0.0
    for pixel_index in range(0, pixel_count, step):
        alpha = float(pixel_values[pixel_index * image.channels + 3])
        minimum = min(minimum, alpha)
        maximum = max(maximum, alpha)
    return minimum < 0.999 and maximum - minimum > 0.0001


def material_uses_color_as_alpha(material: bpy.types.Material) -> bool:
    if not material.use_nodes or material.node_tree is None:
        return False
    return any(
        link.from_socket.name == "Color"
        and link.to_socket.name == "Alpha"
        and link.to_node.type == "BSDF_PRINCIPLED"
        for link in material.node_tree.links
    )


def rebuild_surface_material(
    material: bpy.types.Material,
    color_path: Path,
    normal_path: Path,
    specular_path: Path,
) -> dict[str, str]:
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    color = new_image_node(nodes, color_path, "sRGB", "official_color")
    normal = new_image_node(nodes, normal_path, "Non-Color", "official_normal")
    specular = new_image_node(
        nodes, specular_path, "Non-Color", "official_specular"
    )
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 1.0
    shader.inputs["Roughness"].default_value = 0.52

    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    if "Specular IOR Level" in shader.inputs:
        links.new(specular.outputs["Color"], shader.inputs["Specular IOR Level"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return {
        "color": str(color_path.resolve()),
        "normal": str(normal_path.resolve()),
        "specular": str(specular_path.resolve()),
    }


def rebuild_opacity_material(
    material: bpy.types.Material, color_path: Path
) -> dict[str, str]:
    uses_color_alpha = material_uses_color_as_alpha(material)
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    color = new_image_node(nodes, color_path, "sRGB", "official_opacity_color")
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = OPACITY_ROUGHNESS
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.0
    alpha_source = "alpha"
    if uses_color_alpha:
        links.new(color.outputs["Color"], shader.inputs["Alpha"])
        alpha_source = "color_luminance"
    elif image_has_useful_alpha(color.image):
        links.new(color.outputs["Alpha"], shader.inputs["Alpha"])
    else:
        grayscale = nodes.new("ShaderNodeRGBToBW")
        links.new(color.outputs["Color"], grayscale.inputs["Color"])
        links.new(grayscale.outputs["Val"], shader.inputs["Alpha"])
        alpha_source = "grayscale"
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.surface_render_method = "DITHERED"
    material.use_transparency_overlap = False
    return {
        "color": str(color_path.resolve()),
        "alpha_source": alpha_source,
    }


def reconnect_official_materials(
    avatar: ImportedAvatar, texture_dir: Path, texture_prefix: str
) -> dict[str, dict[str, str]]:
    texture_dir = texture_dir.resolve()
    materials = {slot.material.name: slot.material for slot in avatar.mesh.material_slots}
    provenance: dict[str, dict[str, str]] = {}
    for suffix in ("body", "head"):
        material_name = f"{texture_prefix}_{suffix}"
        provenance[material_name] = rebuild_surface_material(
            materials[material_name],
            require_texture(texture_dir, f"{texture_prefix}_{suffix}_color.tga"),
            require_texture(texture_dir, f"{texture_prefix}_{suffix}_normal.tga"),
            require_texture(texture_dir, f"{texture_prefix}_{suffix}_specular.tga"),
        )
    opacity_name = f"{texture_prefix}_opacity"
    provenance[opacity_name] = rebuild_opacity_material(
        materials[opacity_name],
        require_texture(texture_dir, f"{texture_prefix}_opacity_color.tga"),
    )
    return provenance


def evaluated_bounds(mesh: bpy.types.Object) -> Bounds:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    corners = tuple(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    minimum = Vector(tuple(min(corner[axis] for corner in corners) for axis in range(3)))
    maximum = Vector(tuple(max(corner[axis] for corner in corners) for axis in range(3)))
    return Bounds(minimum=minimum, maximum=maximum, corners=corners)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("rocketbox_review_world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.035, 0.043, 0.052, 1.0)
    background.inputs["Strength"].default_value = 0.16
    scene.view_settings.look = "AgX - Medium High Contrast"


def simple_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.72
    return material


def add_review_floor(bounds: Bounds) -> bpy.types.Object:
    size = max(bounds.dimensions.x, bounds.dimensions.y, 2.5) * 2.4
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(bounds.center.x, bounds.center.y, bounds.minimum.z - 0.003),
    )
    floor = bpy.context.object
    floor.name = "rocketbox_review_floor"
    floor.data.materials.append(simple_material("review_floor", (0.16, 0.18, 0.20, 1.0)))
    return floor


def add_direction_arrow(bounds: Bounds) -> tuple[bpy.types.Object, bpy.types.Object]:
    height = max(bounds.dimensions.z, 1.0)
    length = height * 0.34
    x = bounds.maximum.x - bounds.dimensions.x * 0.08
    z = bounds.minimum.z + 0.008
    red = simple_material("front_minus_y_arrow", (0.86, 0.12, 0.09, 1.0))
    bpy.ops.mesh.primitive_cube_add(
        location=(x, bounds.center.y - length * 0.42, z),
        scale=(height * 0.012, length * 0.38, height * 0.006),
    )
    shaft = bpy.context.object
    shaft.name = "FRONT_-Y_arrow_shaft"
    shaft.data.materials.append(red)
    bpy.ops.mesh.primitive_cone_add(
        vertices=4,
        radius1=height * 0.055,
        radius2=0.0,
        depth=height * 0.11,
        location=(x, bounds.center.y - length * 0.84, z),
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    tip = bpy.context.object
    tip.name = "FRONT_-Y_arrow_tip"
    tip.data.materials.append(red)
    return shaft, tip


def set_review_guides_visibility(view_name: str) -> None:
    floor = bpy.data.objects.get("rocketbox_review_floor")
    if floor is not None:
        floor.hide_render = view_name == "top"
    for arrow_name in ("FRONT_-Y_arrow_shaft", "FRONT_-Y_arrow_tip"):
        arrow = bpy.data.objects.get(arrow_name)
        if arrow is not None:
            arrow.hide_render = view_name != "top"


def add_area_light(name: str, location: Vector, energy: float, size: float, target: Vector) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    look_at(light, target)


def add_lighting(bounds: Bounds) -> None:
    center = bounds.center
    height = max(bounds.dimensions.z, 1.0)
    key_energy, fill_energy, rim_energy = REVIEW_LIGHT_ENERGIES
    add_area_light(
        "review_key",
        center + Vector((-height * 0.8, -height * 1.2, height * 0.9)),
        key_energy,
        height * 0.7,
        center,
    )
    add_area_light(
        "review_fill",
        center + Vector((height * 0.9, -height * 0.5, height * 0.45)),
        fill_energy,
        height * 0.8,
        center,
    )
    add_area_light(
        "review_rim",
        center + Vector((0.0, height * 1.0, height * 0.9)),
        rim_energy,
        height * 0.55,
        center,
    )


def make_camera() -> bpy.types.Object:
    data = bpy.data.cameras.new("rocketbox_review_camera")
    data.type = "ORTHO"
    data.lens = 50.0
    data.clip_start = 0.01
    data.clip_end = 100.0
    camera = bpy.data.objects.new("rocketbox_review_camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def camera_frame_bounds(
    camera: bpy.types.Object, scene: bpy.types.Scene
) -> tuple[float, float, float, float]:
    frame_corners = camera.data.view_frame(scene=scene)
    return (
        min(corner.x for corner in frame_corners),
        max(corner.x for corner in frame_corners),
        min(corner.y for corner in frame_corners),
        max(corner.y for corner in frame_corners),
    )


def set_camera(
    camera: bpy.types.Object,
    target: Vector,
    direction: tuple[float, float, float],
    bounds: Bounds,
    resolution: tuple[int, int],
    fixed_scale: float | None = None,
) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    direction_vector = Vector(direction).normalized()
    distance = max(bounds.dimensions.z, bounds.dimensions.x, 1.0) * 3.2
    camera.location = target + direction_vector * distance
    look_at(camera, target)
    bpy.context.view_layer.update()
    if fixed_scale is None:
        inverse = camera.matrix_world.inverted()
        camera_corners = tuple(inverse @ corner for corner in bounds.corners)
        content_width = max(corner.x for corner in camera_corners) - min(
            corner.x for corner in camera_corners
        )
        content_height = max(corner.y for corner in camera_corners) - min(
            corner.y for corner in camera_corners
        )
        camera.data.ortho_scale = 1.0
        frame_left, frame_right, frame_bottom, frame_top = camera_frame_bounds(
            camera, scene
        )
        frame_width = frame_right - frame_left
        frame_height = frame_top - frame_bottom
        fixed_scale = max(
            content_width / frame_width * BODY_FRAME_MARGIN,
            content_height / frame_height * BODY_FRAME_MARGIN,
            0.25,
        )
    camera.data.ortho_scale = fixed_scale


def emission_material(name: str) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.96, 0.98, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 2.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def legend_text(
    asset_id: str,
    source_name: str,
    view_name: str,
    forward_axis: str,
    up_axis: str,
) -> str:
    direction_suffix = " (RED ARROW)" if view_name == "top" else ""
    return (
        f"{asset_id}\n"
        f"SOURCE {source_name}\n"
        f"{view_name.upper()} | REST POSE / NO ACTION\n"
        f"UP {up_axis} | FRONT {forward_axis}{direction_suffix}"
    )


def annotate_still(
    output_path: Path,
    asset_id: str,
    source_name: str,
    view_name: str,
    forward_axis: str,
    up_axis: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(output_path) as loaded:
        source = loaded.convert("RGBA")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(18, round(source.width * 0.018))
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), font_size)
        if font_path.is_file()
        else ImageFont.load_default()
    )
    text = legend_text(asset_id, source_name, view_name, forward_axis, up_axis)
    spacing = max(3, font_size // 5)
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    padding = max(18, round(source.width * 0.018))
    left = padding
    top = padding
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.rectangle(
        (
            left - padding // 2,
            top - padding // 2,
            left + width + padding // 2,
            top + height + padding // 2,
        ),
        fill=(3, 8, 14, 210),
    )
    draw.multiline_text(
        (left, top),
        text,
        font=font,
        fill=(245, 248, 252, 255),
        spacing=spacing,
    )
    Image.alpha_composite(source, overlay).save(output_path)


def remove_legend() -> None:
    for obj in tuple(bpy.data.objects):
        if obj.name.startswith("rocketbox_review_legend"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                bpy.data.curves.remove(data)


def add_legend(
    camera: bpy.types.Object,
    asset_id: str,
    source_name: str,
    view_name: str,
    forward_axis: str,
    up_axis: str,
) -> None:
    remove_legend()
    curve = bpy.data.curves.new("rocketbox_review_legend_curve", type="FONT")
    curve.body = legend_text(
        asset_id, source_name, view_name, forward_axis, up_axis
    )
    curve.align_x = "LEFT"
    curve.align_y = "TOP"
    curve.size = camera.data.ortho_scale * LEGEND_FONT_SCALE
    curve.space_line = 1.05
    text = bpy.data.objects.new("rocketbox_review_legend", curve)
    bpy.context.collection.objects.link(text)
    text.data.materials.append(emission_material("rocketbox_review_legend_material"))
    text.parent = camera
    scene = bpy.context.scene
    frame_left, frame_right, frame_bottom, frame_top = camera_frame_bounds(
        camera, scene
    )
    margin = camera.data.ortho_scale * LEGEND_MARGIN_SCALE
    text.location = (frame_left + margin, frame_top - margin, -1.0)
    text.rotation_euler = (0.0, 0.0, 0.0)


def render_still(
    output_path: Path,
    camera: bpy.types.Object,
    avatar: ImportedAvatar,
    bounds: Bounds,
    asset_id: str,
    source_name: str,
    view_name: str,
    direction: tuple[float, float, float],
    resolution: tuple[int, int],
    target: Vector | None = None,
    fixed_scale: float | None = None,
    forward_axis: str = "-Y",
    up_axis: str = "+Z",
) -> None:
    del avatar
    set_review_guides_visibility(view_name)
    target = bounds.center if target is None else target
    set_camera(camera, target, direction, bounds, resolution, fixed_scale)
    remove_legend()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    annotate_still(
        output_path,
        asset_id,
        source_name,
        view_name,
        forward_axis,
        up_axis,
    )


def avatar_roots(imported_objects: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    imported = set(imported_objects)
    return [obj for obj in imported_objects if obj.parent not in imported]


def add_turntable_pivot(avatar: ImportedAvatar, bounds: Bounds) -> bpy.types.Object:
    pivot = bpy.data.objects.new("rocketbox_review_turntable", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = (bounds.center.x, bounds.center.y, bounds.minimum.z)
    for root in avatar_roots(avatar.imported_objects):
        world = root.matrix_world.copy()
        root.parent = pivot
        root.matrix_world = world
    pivot.rotation_mode = "XYZ"
    pivot.rotation_euler[2] = 0.0
    pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
    pivot.rotation_euler[2] = math.tau
    pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=97)
    for turntable_curve_set in tuple(bpy.data.actions):
        for curve in turntable_curve_set.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"
    return pivot


def render_turntable(
    output_path: Path,
    camera: bpy.types.Object,
    avatar: ImportedAvatar,
    bounds: Bounds,
    asset_id: str,
    source_name: str,
    forward_axis: str,
    up_axis: str,
) -> None:
    add_turntable_pivot(avatar, bounds)
    set_review_guides_visibility("turntable")
    set_camera(
        camera,
        bounds.center + Vector((0.0, 0.0, bounds.dimensions.z * 0.02)),
        VIEWS["front"],
        bounds,
        VIDEO_SIZE,
    )
    add_legend(
        camera,
        asset_id,
        source_name,
        "turntable",
        forward_axis,
        up_axis,
    )
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 96
    scene.render.fps = 24
    scene.render.resolution_x, scene.render.resolution_y = VIDEO_SIZE
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    bpy.ops.render.render(animation=True)


def render_review(
    args: argparse.Namespace,
    avatar: ImportedAvatar,
    texture_provenance: dict[str, dict[str, str]],
) -> Path:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_scene()
    bounds = evaluated_bounds(avatar.mesh)
    add_review_floor(bounds)
    add_direction_arrow(bounds)
    add_lighting(bounds)
    camera = make_camera()

    for view_name, direction in VIEWS.items():
        render_still(
            output_dir / BODY_VIEW_FILES[view_name],
            camera,
            avatar,
            bounds,
            args.asset_id,
            args.fbx.name,
            view_name,
            direction,
            BODY_VIEW_SIZE,
            forward_axis=args.forward_axis,
            up_axis=args.up_axis,
        )

    height = bounds.dimensions.z
    width = bounds.dimensions.x
    close_specs = {
        "face_close": (
            bounds.center + Vector((0.0, 0.0, height * 0.38)),
            max(height * 0.55, width / (CLOSE_VIEW_SIZE[0] / CLOSE_VIEW_SIZE[1]) * 0.55),
        ),
        "arms_close": (
            bounds.center + Vector((0.0, 0.0, height * 0.18)),
            max(height * 0.46, width / (CLOSE_VIEW_SIZE[0] / CLOSE_VIEW_SIZE[1]) * 1.12),
        ),
        "feet_close": (
            Vector((bounds.center.x, bounds.center.y, bounds.minimum.z + height * 0.12)),
            max(height * 0.32, width * 0.46),
        ),
    }
    for view_name, (target, scale) in close_specs.items():
        render_still(
            output_dir / CLOSE_VIEW_FILES[view_name],
            camera,
            avatar,
            bounds,
            args.asset_id,
            args.fbx.name,
            view_name,
            VIEWS["front"],
            CLOSE_VIEW_SIZE,
            target=target,
            fixed_scale=scale,
            forward_axis=args.forward_axis,
            up_axis=args.up_axis,
        )

    render_turntable(
        output_dir / VIDEO_FILE,
        camera,
        avatar,
        bounds,
        args.asset_id,
        args.fbx.name,
        args.forward_axis,
        args.up_axis,
    )
    manifest = {
        "schema_version": "rocketbox_source_review_render_v1",
        "asset_id": args.asset_id,
        "blender_version": bpy.app.version_string,
        "source_fbx": str(args.fbx.resolve()),
        "source_fbx_sha256": sha256_file(args.fbx),
        "mesh_name": avatar.mesh.name,
        "vertex_count": len(avatar.mesh.data.vertices),
        "polygon_count": len(avatar.mesh.data.polygons),
        "uv_layer_count": len(avatar.mesh.data.uv_layers),
        "armature_name": avatar.armature.name,
        "bone_count": len(avatar.armature.data.bones),
        "material_slot_names": list(avatar.material_slot_names),
        "material_textures": texture_provenance,
        "view_files": {
            **BODY_VIEW_FILES,
            **CLOSE_VIEW_FILES,
        },
        "video_file": VIDEO_FILE,
        "forward_axis": args.forward_axis,
        "up_axis": args.up_axis,
        "animation_attached": False,
        "pose_position": avatar.armature.data.pose_position,
        "frame_count": 96,
        "fps": 24,
    }
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.forward_axis != "-Y" or args.up_axis != "+Z":
        raise ValueError("Rocketbox source review is pinned to FRONT -Y and UP +Z")
    texture_prefix = ASSET_PREFIXES[args.asset_id]
    clear_scene()
    avatar = import_avatar(args.fbx.resolve(), texture_prefix)
    texture_provenance = reconnect_official_materials(
        avatar, args.texture_dir, texture_prefix
    )
    render_review(args, avatar, texture_provenance)
    print(
        f"ROCKETBOX_SOURCE_REVIEW_RENDER_OK asset_id={args.asset_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
