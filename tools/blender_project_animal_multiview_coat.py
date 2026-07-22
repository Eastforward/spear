#!/usr/bin/env python3
"""Project an edited four-view coat back to the unchanged animal UV.

The edit is represented as a spatial, per-view log-colour ratio between the
fixed source views and their edited counterparts.  Ratios are visibility and
normal weighted on the rest mesh, multiplied with the original Base Color at
each UV corner, and baked to a new atlas.  Geometry, skin weights, skeleton,
and Walk/Idle actions are never regenerated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import struct
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
import numpy as np


SCHEMA = "avengine_generated_animal_multiview_coat_projection_v2"
VIEW_ORDER = ("front", "back", "left", "right")


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--source-view-dir", type=Path, required=True)
    parser.add_argument("--edited-view-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="animal_coat",
        help="Portable lowercase identifier used in output texture and GLB filenames.",
    )
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--visibility-tolerance-ratio", type=float, default=0.003)
    parser.add_argument("--minimum-direct-coverage", type=float, default=0.60)
    parser.add_argument("--luminance-transfer-strength", type=float, default=0.15)
    parser.add_argument(
        "--colour-transfer-mode",
        choices=("relative_chroma", "absolute_edited_chroma"),
        default="relative_chroma",
        help=(
            "relative_chroma applies the edited/source chroma delta to the original "
            "texture; absolute_edited_chroma uses the edited views as the coat-colour "
            "authority while retaining the original texture luminance."
        ),
    )
    parser.add_argument(
        "--absolute-chroma-strength",
        type=float,
        default=1.0,
        help=(
            "Blend strength in [0,1] for absolute_edited_chroma: 0 retains the "
            "original coat chroma and 1 fully follows the edited reference chroma."
        ),
    )
    return parser.parse_args(argv)


def require_new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace output root: {path}")
    path.mkdir(parents=True)
    return path


def load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def srgb_to_linear(values):
    """Decode stored Base Color/image-edit RGB into Blender scene-linear RGB."""
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


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


def image_array(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)
    pixels[..., :3] = srgb_to_linear(pixels[..., :3])
    return image, pixels


def bilinear(image, xy):
    height, width = image.shape[:2]
    x = np.clip(xy[:, 0], 0.0, width - 1.0)
    y = np.clip(xy[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0, :3] * (1.0 - wx) + image[y0, x1, :3] * wx
    bottom = image[y1, x0, :3] * (1.0 - wx) + image[y1, x1, :3] * wx
    return top * (1.0 - wy) + bottom * wy


def upstream_image_nodes(socket):
    result = []
    visited = set()

    def walk(input_socket):
        for link in input_socket.links:
            node = link.from_node
            if node.as_pointer() in visited:
                continue
            visited.add(node.as_pointer())
            if node.type == "TEX_IMAGE" and node.image is not None:
                result.append(node)
            for upstream in node.inputs:
                if upstream.is_linked:
                    walk(upstream)

    walk(socket)
    return result


def base_colour_binding(mesh):
    materials = [slot.material for slot in mesh.material_slots if slot.material]
    if len(materials) != 1:
        raise RuntimeError(f"coat projection currently requires one material, got {len(materials)}")
    material = materials[0]
    if not material.use_nodes or material.node_tree is None:
        raise RuntimeError("generated animal material has no node graph")
    principled = [node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"]
    if len(principled) != 1:
        raise RuntimeError(f"expected one Principled BSDF, got {len(principled)}")
    nodes = upstream_image_nodes(principled[0].inputs["Base Color"])
    unique_images = {node.image.as_pointer(): node.image for node in nodes}
    if len(unique_images) != 1:
        raise RuntimeError(f"expected one Base Color image, got {len(unique_images)}")
    return material, principled[0], nodes, next(iter(unique_images.values()))


def world_geometry(mesh):
    matrix = mesh.matrix_world
    normal_matrix = matrix.to_3x3().inverted().transposed()
    vertices = np.array([tuple(matrix @ item.co) for item in mesh.data.vertices], dtype=np.float64)
    normals = np.array(
        [tuple((normal_matrix @ item.normal).normalized()) for item in mesh.data.vertices],
        dtype=np.float64,
    )
    polygons = [tuple(poly.vertices) for poly in mesh.data.polygons]
    return vertices, normals, polygons


def camera_from_contract(name, view, ortho_scale):
    data = bpy.data.cameras.new(f"ProjectionCamera_{name}")
    data.type = "ORTHO"
    data.ortho_scale = float(ortho_scale)
    camera = bpy.data.objects.new(f"ProjectionCamera_{name}", data)
    bpy.context.collection.objects.link(camera)
    camera.location = Vector(view["camera_location"])
    direction = Vector(view["camera_direction_to_subject"]).normalized()
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    return camera, direction


def vertex_log_ratios(
    mesh,
    vertices,
    normals,
    manifest,
    source_dir,
    edited_dir,
    tolerance,
    minimum_direct_coverage,
    luminance_transfer_strength,
    colour_transfer_mode,
):
    bvh = BVHTree.FromPolygons(
        [tuple(item) for item in vertices],
        [tuple(poly.vertices) for poly in mesh.data.polygons],
        all_triangles=False,
    )
    accumulation = np.zeros((len(vertices), 3), dtype=np.float64)
    weights = np.zeros(len(vertices), dtype=np.float64)
    per_view = {}
    cameras = []
    scene = bpy.context.scene

    for name in VIEW_ORDER:
        source_handle, source = image_array(source_dir / f"{name}.png")
        edited_handle, edited = image_array(edited_dir / f"{name}.png")
        if source.shape != edited.shape or source.shape[:2] != (512, 512):
            raise RuntimeError(f"source/edited view mismatch for {name}")
        camera, direction = camera_from_contract(
            name, manifest["views"][name], manifest["ortho_scale"]
        )
        cameras.append(camera)
        projection = np.array(
            [tuple(world_to_camera_view(scene, camera, Vector(item))) for item in vertices],
            dtype=np.float64,
        )
        in_frame = (
            (projection[:, 0] >= 0.0)
            & (projection[:, 0] <= 1.0)
            & (projection[:, 1] >= 0.0)
            & (projection[:, 1] <= 1.0)
            & (projection[:, 2] >= 0.0)
        )
        facing = normals @ (-np.asarray(direction, dtype=np.float64))
        candidates = np.flatnonzero(in_frame & (facing > 0.08))
        visible = []
        ray_distance = float(manifest["camera_distance"]) * 2.0
        for index in candidates:
            point = Vector(vertices[index])
            origin = point - direction * ray_distance
            hit, _, _, _ = bvh.ray_cast(origin, direction, ray_distance + tolerance * 2.0)
            if hit is not None and (hit - point).length <= tolerance:
                visible.append(int(index))
        visible = np.asarray(visible, dtype=np.int64)
        if not len(visible):
            raise RuntimeError(f"no visible projected vertices for {name}")
        height, width = source.shape[:2]
        xy = np.column_stack(
            (projection[visible, 0] * (width - 1), projection[visible, 1] * (height - 1))
        )
        source_rgb = np.clip(bilinear(source, xy), 0.0, 1.0)
        edited_rgb = np.clip(bilinear(edited, xy), 0.0, 1.0)
        source_log = np.log(source_rgb + 0.025)
        edited_log = np.log(edited_rgb + 0.025)
        source_luminance = source_log.mean(axis=1, keepdims=True)
        edited_luminance = edited_log.mean(axis=1, keepdims=True)
        source_chroma = source_log - source_luminance
        edited_chroma = edited_log - edited_luminance
        if colour_transfer_mode == "relative_chroma":
            colour_field = (
                edited_chroma
                - source_chroma
                + luminance_transfer_strength * (edited_luminance - source_luminance)
            )
        else:
            # Generated edit views are the breed/coat-colour authority.  Their
            # spatial log chroma is transferred directly, while the original
            # UV texture remains the luminance/detail authority in main().
            colour_field = edited_chroma
        colour_field = np.clip(colour_field, math.log(0.18), math.log(5.5))
        view_weight = np.square(np.clip(facing[visible], 0.0, 1.0))
        accumulation[visible] += colour_field * view_weight[:, None]
        weights[visible] += view_weight
        per_view[name] = {
            "candidate_vertex_count": int(len(candidates)),
            "visible_vertex_count": int(len(visible)),
            "mean_facing_weight": float(view_weight.mean()),
        }
        bpy.data.images.remove(source_handle)
        bpy.data.images.remove(edited_handle)

    covered = weights > 1.0e-8
    covered_count = int(np.count_nonzero(covered))
    coverage_ratio = covered_count / len(vertices)
    if coverage_ratio < minimum_direct_coverage:
        raise RuntimeError(
            "four-view projection missed the required direct coverage: "
            f"covered={covered_count}/{len(vertices)} ratio={coverage_ratio:.6f} "
            f"required={minimum_direct_coverage:.6f} "
            f"per_view={per_view}"
        )
    colour_field = np.zeros_like(accumulation)
    colour_field[covered] = accumulation[covered] / weights[covered, None]

    missing = np.flatnonzero(~covered)
    if len(missing):
        tree = KDTree(int(np.count_nonzero(covered)))
        for slot, index in enumerate(np.flatnonzero(covered)):
            tree.insert(Vector(vertices[index]), slot)
        tree.balance()
        covered_indices = np.flatnonzero(covered)
        for index in missing:
            _, slot, _ = tree.find(Vector(vertices[index]))
            colour_field[index] = colour_field[covered_indices[slot]]
    return colour_field.astype(np.float32), covered, per_view, cameras


def sample_base_texture_for_loops(mesh, image):
    uv_layer = mesh.data.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("generated animal has no active UV layer")
    loop_count = len(mesh.data.loops)
    uv = np.empty(loop_count * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], 4)
    pixels[..., :3] = srgb_to_linear(pixels[..., :3])
    xy = np.column_stack(
        (uv[:, 0] * (image.size[0] - 1), uv[:, 1] * (image.size[1] - 1))
    )
    return bilinear(pixels, xy)


def bake_corner_colours(mesh, corner_rgb, output_png, texture_size):
    attribute = mesh.data.color_attributes.get("ProjectedCoat")
    if attribute is not None:
        mesh.data.color_attributes.remove(attribute)
    attribute = mesh.data.color_attributes.new(
        name="ProjectedCoat", type="FLOAT_COLOR", domain="CORNER"
    )
    rgba = np.column_stack((np.clip(corner_rgb, 0.0, 1.0), np.ones(len(corner_rgb))))
    attribute.data.foreach_set("color", rgba.astype(np.float32).reshape(-1))

    original_materials = [slot.material for slot in mesh.material_slots]
    original_indices = [poly.material_index for poly in mesh.data.polygons]
    bake_material = bpy.data.materials.new("ProjectedCoatBakeMaterial")
    bake_material.use_nodes = True
    nodes = bake_material.node_tree.nodes
    nodes.clear()
    vertex = nodes.new("ShaderNodeVertexColor")
    vertex.layer_name = attribute.name
    emission = nodes.new("ShaderNodeEmission")
    target = nodes.new("ShaderNodeTexImage")
    output = nodes.new("ShaderNodeOutputMaterial")
    bake_material.node_tree.links.new(vertex.outputs["Color"], emission.inputs["Color"])
    bake_material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    image = bpy.data.images.new(
        "ProjectedAnimalCoatBaseColor",
        width=texture_size,
        height=texture_size,
        alpha=True,
        float_buffer=False,
    )
    image.colorspace_settings.name = "sRGB"
    target.image = image
    target.select = True
    nodes.active = target

    mesh.data.materials.clear()
    mesh.data.materials.append(bake_material)
    for poly in mesh.data.polygons:
        poly.material_index = 0
    bpy.ops.object.select_all(action="DESELECT")
    mesh.hide_set(False)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.bake.margin = 20
    bpy.ops.object.bake(type="EMIT", margin=20)
    image.filepath_raw = str(output_png)
    image.file_format = "PNG"
    image.save()

    mesh.data.materials.clear()
    for material in original_materials:
        mesh.data.materials.append(material)
    for poly, material_index in zip(mesh.data.polygons, original_indices):
        poly.material_index = material_index
    return image


def glb_json(path: Path):
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise RuntimeError("invalid exported GLB")
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    if json_type != 0x4E4F534A:
        raise RuntimeError("exported GLB has no JSON chunk")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def main():
    args = parse_argv()
    if args.texture_size not in {512, 1024, 2048, 4096}:
        raise RuntimeError("--texture-size must be 512, 1024, 2048, or 4096")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_stem) is None:
        raise RuntimeError("--output-stem must match [a-z0-9][a-z0-9_-]{0,63}")
    if not 0.40 <= args.minimum_direct_coverage <= 0.95:
        raise RuntimeError("--minimum-direct-coverage must be in [0.40, 0.95]")
    if not 0.0 <= args.luminance_transfer_strength <= 1.0:
        raise RuntimeError("--luminance-transfer-strength must be in [0, 1]")
    if not 0.0 <= args.absolute_chroma_strength <= 1.0:
        raise RuntimeError("--absolute-chroma-strength must be in [0, 1]")
    input_glb = args.input_glb.resolve()
    source_dir = args.source_view_dir.resolve()
    edited_dir = args.edited_view_dir.resolve()
    output_root = require_new_directory(args.output_root)
    view_manifest = load_json(source_dir / "render_manifest.json")
    if (
        view_manifest.get("schema") != "avengine_generated_animal_coat_views_v1"
        or view_manifest.get("input_glb") != str(input_glb)
        or view_manifest.get("view_order") != list(VIEW_ORDER)
        or view_manifest.get("rest_pose") is not True
        or view_manifest.get("resolution") != [512, 512]
    ):
        raise RuntimeError("source view contract does not match this animal")
    for directory in (source_dir, edited_dir):
        for name in VIEW_ORDER:
            path = directory / f"{name}.png"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"missing projection input: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    mesh = primary_skinned_mesh()
    for armature in [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]:
        armature.data.pose_position = "REST"
    for other in [item for item in bpy.context.scene.objects if item.type == "MESH"]:
        other.hide_render = other != mesh
    bpy.context.view_layer.update()

    material, principled, base_nodes, original_base_image = base_colour_binding(mesh)
    vertices, normals, polygons = world_geometry(mesh)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tolerance = diagonal * args.visibility_tolerance_ratio
    colour_field, covered, per_view, cameras = vertex_log_ratios(
        mesh,
        vertices,
        normals,
        view_manifest,
        source_dir,
        edited_dir,
        tolerance,
        args.minimum_direct_coverage,
        args.luminance_transfer_strength,
        args.colour_transfer_mode,
    )

    loop_vertex = np.empty(len(mesh.data.loops), dtype=np.int32)
    mesh.data.loops.foreach_get("vertex_index", loop_vertex)
    base_rgb = sample_base_texture_for_loops(mesh, original_base_image)
    if args.colour_transfer_mode == "relative_chroma":
        candidate_rgb = base_rgb * np.exp(colour_field[loop_vertex])
    else:
        base_log = np.log(base_rgb + 0.025)
        base_geometric_luminance = base_log.mean(axis=1, keepdims=True)
        base_chroma = base_log - base_geometric_luminance
        blended_chroma = (
            (1.0 - args.absolute_chroma_strength) * base_chroma
            + args.absolute_chroma_strength * colour_field[loop_vertex]
        )
        candidate_rgb = np.maximum(
            np.exp(base_geometric_luminance + blended_chroma) - 0.025,
            0.0,
        )
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    base_luminance = base_rgb @ rec709
    candidate_luminance = candidate_rgb @ rec709
    luminance_normalization = (base_luminance + 1.0e-5) / (
        candidate_luminance + 1.0e-5
    )
    corner_rgb = np.clip(candidate_rgb * luminance_normalization[:, None], 0.0, 1.0)
    texture_path = output_root / f"{args.output_stem}_base_color.png"
    baked_image = bake_corner_colours(mesh, corner_rgb, texture_path, args.texture_size)

    for node in base_nodes:
        node.image = baked_image
    for socket in (principled.inputs.get("Metallic"), principled.inputs.get("Roughness")):
        if socket is not None:
            for link in list(socket.links):
                material.node_tree.links.remove(link)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.82
    for camera in cameras:
        bpy.data.objects.remove(camera, do_unlink=True)
    for armature in [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]:
        armature.data.pose_position = "POSE"
    for action in bpy.data.actions:
        if action.name.startswith("Walking"):
            action.name = "Walking"
        elif action.name.startswith("Idle"):
            action.name = "Idle"

    output_glb = output_root / f"animated_walk_idle_{args.output_stem}.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_morph=True,
        export_apply=False,
    )
    document = glb_json(output_glb)
    animations = sorted(item.get("name") for item in document.get("animations", []))
    primitives = [
        primitive
        for gltf_mesh in document.get("meshes", [])
        for primitive in gltf_mesh.get("primitives", [])
    ]
    skinned = [
        primitive
        for primitive in primitives
        if {"JOINTS_0", "WEIGHTS_0"}.issubset(primitive.get("attributes", {}))
    ]
    if animations != ["Idle", "Walking"] or len(document.get("skins", [])) != 1 or len(skinned) != 1:
        raise RuntimeError(
            f"output lost animation/skin contract: animations={animations} "
            f"skins={len(document.get('skins', []))} skinned={len(skinned)}"
        )

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "technical_spike_only",
        "formal_dataset_registration_authorized": False,
        "output_stem": args.output_stem,
        "input_glb": str(input_glb),
        "source_view_dir": str(source_dir),
        "edited_view_dir": str(edited_dir),
        "projection_method": "visibility_weighted_multiview_log_chroma_linear_luminance_preserved_v5",
        "not_global_rgb_factor": True,
        "colour_transfer_mode": args.colour_transfer_mode,
        "absolute_chroma_strength": args.absolute_chroma_strength,
        "explicit_srgb_to_linear_input_decode": True,
        "bake_output_colourspace": "sRGB",
        "luminance_transfer_strength": args.luminance_transfer_strength,
        "per_uv_corner_rec709_linear_luminance_preserved": True,
        "geometry_skin_skeleton_and_actions_preserved_by_design": True,
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "loop_count": len(mesh.data.loops),
        "directly_covered_vertex_count": int(np.count_nonzero(covered)),
        "nearest_filled_vertex_count": int(np.count_nonzero(~covered)),
        "direct_coverage_ratio": float(np.mean(covered)),
        "minimum_direct_coverage_required": args.minimum_direct_coverage,
        "visibility_tolerance": tolerance,
        "per_view": per_view,
        "texture_size": args.texture_size,
        "base_color_texture": str(texture_path),
        "output_glb": str(output_glb),
        "readback": {
            "animations": animations,
            "skin_count": len(document.get("skins", [])),
            "skinned_primitive_count": len(skinned),
        },
        "material_policy": {"metallic": 0.0, "roughness": 0.82},
        "next_gate": "walking_visual_coat_and_deformation_review",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "GENERATED_ANIMAL_MULTIVIEW_COAT_PROJECT_OK "
        f"coverage={manifest['direct_coverage_ratio']:.6f} output={output_root}"
    )


if __name__ == "__main__":
    main()
