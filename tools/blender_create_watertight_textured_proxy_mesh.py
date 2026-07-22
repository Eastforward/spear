"""Build a watertight, textured runtime proxy from an image-to-3D GLB.

The source GLB remains the appearance and surface authority.  A voxel remesh
removes cracks, non-manifold scraps, and zero-thickness ribbons; shrinkwrap
then returns the proxy to the source surface.  UVs, corner colors, and
materials are transferred from the source after topology regularization.

This tool intentionally exports a static mesh.  Rigging and animation are a
separate, auditable stage so a failed topology experiment cannot modify an
approved skeleton or action.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


SCHEMA = "avengine_watertight_textured_runtime_proxy_v1"


def stage(label, started_at=None, **values):
    """Publish long Blender phases so a healthy CPU job never looks hung."""
    suffix = " ".join(f"{key}={value}" for key, value in values.items())
    if started_at is not None:
        suffix = f"elapsed_s={time.perf_counter() - started_at:.2f} {suffix}".strip()
    print(f"WATERTIGHT_PROXY_STAGE {label} {suffix}".rstrip(), flush=True)
    return time.perf_counter()


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--attribute-source",
        type=Path,
        help=(
            "Optional lower-face PBR copy used only for UV/color/material transfer. "
            "The full-resolution --source remains the geometry and shrinkwrap authority."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--voxel-resolution", type=int, default=220)
    parser.add_argument("--target-faces", type=int, default=100000)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument(
        "--shrinkwrap-strength",
        type=float,
        default=1.0,
        help=(
            "Blend between the clean voxel surface (0) and the nearest raw "
            "I23D surface (1). Values below 1 prevent closed proxies from "
            "copying deep crack-like source folds."
        ),
    )
    parser.add_argument(
        "--post-shrinkwrap-smooth-iterations",
        type=int,
        default=0,
        help=(
            "Small volume-preserving cleanup after returning to the source "
            "surface. Use this to remove closed but visually crack-like I23D "
            "folds before PBR bake and rigging."
        ),
    )
    parser.add_argument(
        "--torso-fold-repair-iterations",
        type=int,
        default=0,
        help=(
            "Optional weighted Laplacian cleanup restricted to the normalized "
            "mid-torso. This removes closed crack-like abdomen folds without "
            "smoothing paws, head, or tail."
        ),
    )
    parser.add_argument(
        "--attribute-transfer-backend",
        choices=("bake", "bvh", "data-transfer"),
        default="bake",
        help=(
            "Bake a fresh PBR atlas (production default), use the diagnostic BVH "
            "sampler, or retain Blender's slower modifier backend."
        ),
    )
    parser.add_argument("--bake-resolution", type=int, default=2048)
    parser.add_argument(
        "--base-color-encoding-policy",
        choices=("preserve-bake", "srgb-to-linear"),
        default="preserve-bake",
    )
    parser.add_argument(
        "--base-color-gain",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--double-sided", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def real_meshes():
    return [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and len(item.data.polygons) > 0
    ]


def activate(obj):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transforms(obj):
    activate(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def mesh_diagonal(obj):
    points = [vertex.co for vertex in obj.data.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return sum((maximum[axis] - minimum[axis]) ** 2 for axis in range(3)) ** 0.5


def topology_stats(obj):
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        boundary = 0
        wire = 0
        over_two = 0
        noncontiguous = 0
        for edge in bm.edges:
            linked = len(edge.link_faces)
            if linked == 0:
                wire += 1
            elif linked == 1:
                boundary += 1
            elif linked > 2:
                over_two += 1
            elif not edge.is_contiguous:
                noncontiguous += 1
        return {
            "vertices": len(bm.verts),
            "edges": len(bm.edges),
            "faces": len(bm.faces),
            "boundary_edges": boundary,
            "wire_edges": wire,
            "nonmanifold_edges_over_two_faces": over_two,
            "noncontiguous_two_face_edges": noncontiguous,
        }
    finally:
        bm.free()


def voxel_remesh(proxy, voxel_size, smooth_iterations):
    activate(proxy)
    proxy.data.remesh_voxel_size = voxel_size
    proxy.data.remesh_voxel_adaptivity = 0.0
    proxy.data.use_remesh_preserve_volume = True
    bpy.ops.object.voxel_remesh()
    for _index in range(smooth_iterations):
        modifier = proxy.modifiers.new(name="WatertightLaplacian", type="LAPLACIANSMOOTH")
        modifier.iterations = 1
        modifier.lambda_factor = 0.12
        modifier.use_volume_preserve = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def post_shrinkwrap_smooth(proxy, iterations):
    activate(proxy)
    for index in range(iterations):
        modifier = proxy.modifiers.new(
            name=f"PostShrinkwrapCleanup{index:02d}",
            type="LAPLACIANSMOOTH",
        )
        modifier.iterations = 1
        modifier.lambda_factor = 0.08
        modifier.use_volume_preserve = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def repair_normalized_torso_folds(proxy, iterations):
    if iterations == 0:
        return {
            "iterations": 0,
            "selected_vertices": 0,
            "policy": "disabled",
        }
    points = np.asarray(
        [tuple(vertex.co) for vertex in proxy.data.vertices],
        dtype=np.float64,
    )
    minimum = points.min(axis=0)
    extent = np.ptp(points, axis=0)
    normalized = (points - minimum) / np.maximum(extent, 1.0e-9)
    longitudinal_axis = 0 if extent[0] >= extent[1] else 1
    longitudinal = normalized[:, longitudinal_axis]
    vertical = normalized[:, 2]

    def tapered(values, lower, upper, fade):
        enter = np.clip((values - lower) / fade, 0.0, 1.0)
        leave = np.clip((upper - values) / fade, 0.0, 1.0)
        return np.minimum(enter, leave)

    weights = tapered(longitudinal, 0.25, 0.70, 0.08)
    weights *= tapered(vertical, 0.34, 0.72, 0.08)
    selected = np.flatnonzero(weights > 1.0e-4)
    if len(selected) == 0:
        raise RuntimeError("normalized torso-fold repair selected no vertices")
    group = proxy.vertex_groups.new(name="NormalizedTorsoFoldRepair")
    group_name = group.name
    for index in selected:
        group.add([int(index)], float(weights[index]), "REPLACE")
    activate(proxy)
    modifier = proxy.modifiers.new(
        name="NormalizedTorsoFoldCleanup",
        type="LAPLACIANSMOOTH",
    )
    modifier.vertex_group = group_name
    modifier.iterations = iterations
    modifier.lambda_factor = 0.18
    modifier.use_volume_preserve = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    remaining_group = proxy.vertex_groups.get(group_name)
    if remaining_group is not None:
        proxy.vertex_groups.remove(remaining_group)
    return {
        "iterations": iterations,
        "selected_vertices": int(len(selected)),
        "longitudinal_axis": int(longitudinal_axis),
        "normalized_longitudinal_range": [0.25, 0.70],
        "normalized_vertical_range": [0.34, 0.72],
        "fade": 0.08,
        "lambda_factor": 0.18,
        "policy": "weighted_mid_torso_only_preserve_volume",
    }


def shrinkwrap_to_source(proxy, source, strength):
    clean_positions = [vertex.co.copy() for vertex in proxy.data.vertices]
    activate(proxy)
    modifier = proxy.modifiers.new(name="ReturnToPBRSurface", type="SHRINKWRAP")
    modifier.target = source
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.wrap_mode = "ON_SURFACE"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    if strength < 1.0:
        for vertex, clean in zip(proxy.data.vertices, clean_positions):
            vertex.co = clean.lerp(vertex.co, strength)
        proxy.data.update()


def decimate(proxy, target_faces):
    current = len(proxy.data.polygons)
    if current <= target_faces:
        return current
    activate(proxy)
    modifier = proxy.modifiers.new(name="RuntimeFaceBudget", type="DECIMATE")
    modifier.ratio = float(target_faces) / float(current)
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return len(proxy.data.polygons)


def prepare_surface_materials(proxy, source):
    proxy.data.materials.clear()
    for material in source.data.materials:
        proxy.data.materials.append(material)


def transfer_surface_attributes_modifier(proxy, source):
    prepare_surface_materials(proxy, source)
    activate(proxy)
    modifier = proxy.modifiers.new(name="TransferPBRSurfaceAttributes", type="DATA_TRANSFER")
    modifier.object = source
    modifier.use_loop_data = True
    available = {"UV"}
    if source.data.color_attributes:
        available.add("COLOR_CORNER")
    modifier.data_types_loops = available
    modifier.loop_mapping = "POLYINTERP_NEAREST"
    print("WATERTIGHT_PROXY_STAGE surface_transfer_modifier_layout_start", flush=True)
    bpy.ops.object.datalayout_transfer(modifier=modifier.name)
    print("WATERTIGHT_PROXY_STAGE surface_transfer_modifier_layout_done", flush=True)
    print("WATERTIGHT_PROXY_STAGE surface_transfer_modifier_apply_start", flush=True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    print("WATERTIGHT_PROXY_STAGE surface_transfer_modifier_apply_done", flush=True)
    if not proxy.data.uv_layers:
        raise RuntimeError("PBR UV transfer produced no UV layer")
    for polygon in proxy.data.polygons:
        polygon.use_smooth = True
    return {
        "backend": "data-transfer",
        "uv_layers": [layer.name for layer in proxy.data.uv_layers],
        "color_attributes": [attribute.name for attribute in proxy.data.color_attributes],
        "material_slots": [material.name if material else None for material in proxy.data.materials],
    }


def barycentric_weights(point, first, second, third):
    edge_zero = second - first
    edge_one = third - first
    relative = point - first
    dot_zero_zero = edge_zero.dot(edge_zero)
    dot_zero_one = edge_zero.dot(edge_one)
    dot_one_one = edge_one.dot(edge_one)
    dot_relative_zero = relative.dot(edge_zero)
    dot_relative_one = relative.dot(edge_one)
    denominator = dot_zero_zero * dot_one_one - dot_zero_one * dot_zero_one
    if abs(denominator) <= 1e-20:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    second_weight = (
        dot_one_one * dot_relative_zero - dot_zero_one * dot_relative_one
    ) / denominator
    third_weight = (
        dot_zero_zero * dot_relative_one - dot_zero_one * dot_relative_zero
    ) / denominator
    first_weight = 1.0 - second_weight - third_weight
    return (first_weight, second_weight, third_weight)


def weighted_vector(values, weights):
    result = Vector((0.0,) * len(values[0]))
    for value, weight in zip(values, weights):
        result += Vector(value) * weight
    return result


def transfer_surface_attributes_bvh(proxy, source):
    """Transfer PBR corner data by casting every face corner inward."""
    prepare_surface_materials(proxy, source)
    source.data.calc_loop_triangles()
    triangles = list(source.data.loop_triangles)
    source_points = [vertex.co.copy() for vertex in source.data.vertices]
    bvh = BVHTree.FromPolygons(
        source_points,
        [tuple(triangle.vertices) for triangle in triangles],
        all_triangles=True,
    )
    if bvh is None or not triangles:
        raise RuntimeError("could not build PBR attribute BVH")
    source_uv = source.data.uv_layers.active
    if source_uv is None:
        raise RuntimeError("PBR attribute source has no active UV layer")

    proxy.data.update()
    ray_offset = max(mesh_diagonal(proxy) * 0.02, 1e-6)
    samples = {}
    ray_hit_count = 0
    nearest_fallback_count = 0
    total_corners = len(proxy.data.loops)
    processed_corners = 0
    for polygon in proxy.data.polygons:
        outward = polygon.normal.normalized()
        for loop_index in polygon.loop_indices:
            loop = proxy.data.loops[loop_index]
            point = proxy.data.vertices[loop.vertex_index].co
            nearest = None
            triangle_index = None
            if outward.length_squared > 0.0:
                nearest, _normal, triangle_index, _distance = bvh.ray_cast(
                    point + outward * ray_offset,
                    -outward,
                    ray_offset * 2.5,
                )
            if nearest is not None and triangle_index is not None:
                ray_hit_count += 1
            else:
                nearest, _normal, triangle_index, _distance = bvh.find_nearest(point)
                nearest_fallback_count += 1
            if nearest is None or triangle_index is None:
                raise RuntimeError(
                    f"PBR BVH query failed for proxy loop {loop_index}"
                )
            triangle = triangles[triangle_index]
            weights = barycentric_weights(
                nearest,
                *(source_points[index] for index in triangle.vertices),
            )
            samples[loop_index] = (triangle, weights)
            processed_corners += 1
            if processed_corners % 50000 == 0 or processed_corners == total_corners:
                print(
                    "WATERTIGHT_PROXY_STAGE surface_transfer_bvh_progress "
                    f"corners={processed_corners}/{total_corners}",
                    flush=True,
                )

    while proxy.data.uv_layers:
        proxy.data.uv_layers.remove(proxy.data.uv_layers[0])
    target_uv = proxy.data.uv_layers.new(name=source_uv.name)
    for loop in proxy.data.loops:
        triangle, weights = samples[loop.index]
        uv_values = [
            source_uv.data[loop_index].uv
            for loop_index in triangle.loops
        ]
        target_uv.data[loop.index].uv = weighted_vector(uv_values, weights)

    while proxy.data.color_attributes:
        proxy.data.color_attributes.remove(proxy.data.color_attributes[0])
    skipped_color_attributes = []
    for source_attribute in source.data.color_attributes:
        if source_attribute.domain != "CORNER":
            skipped_color_attributes.append(source_attribute.name)
            continue
        target_attribute = proxy.data.color_attributes.new(
            name=source_attribute.name,
            type=source_attribute.data_type,
            domain="CORNER",
        )
        for loop in proxy.data.loops:
            triangle, weights = samples[loop.index]
            color_values = [
                source_attribute.data[loop_index].color
                for loop_index in triangle.loops
            ]
            target_attribute.data[loop.index].color = weighted_vector(
                color_values,
                weights,
            )

    for polygon in proxy.data.polygons:
        polygon.use_smooth = True
    return {
        "backend": "bvh",
        "bvh_query_count": total_corners,
        "query_domain": "face_corner",
        "outward_ray_hit_count": ray_hit_count,
        "nearest_fallback_count": nearest_fallback_count,
        "outward_ray_offset": ray_offset,
        "source_triangle_count": len(triangles),
        "uv_layers": [layer.name for layer in proxy.data.uv_layers],
        "color_attributes": [
            attribute.name for attribute in proxy.data.color_attributes
        ],
        "skipped_non_corner_color_attributes": skipped_color_attributes,
        "material_slots": [
            material.name if material else None
            for material in proxy.data.materials
        ],
    }


def smart_unwrap(proxy):
    activate(proxy)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=1.1519173063162575,
        island_margin=0.01,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def new_bake_image(name, resolution, color_space):
    image = bpy.data.images.new(
        name=name,
        width=resolution,
        height=resolution,
        alpha=False,
        float_buffer=False,
    )
    image.colorspace_settings.name = color_space
    return image


def reinterpret_baked_srgb_values_as_scene_linear(image):
    """Undo Cycles selected-to-active storing encoded sRGB as linear values."""
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))
    encoded = np.clip(pixels[:, :3], 0.0, 1.0)
    pixels[:, :3] = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()


def apply_base_color_gain(image, gain):
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))
    pixels[:, :3] = np.clip(
        pixels[:, :3] * np.asarray(gain, dtype=np.float32),
        0.0,
        1.0,
    )
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()


def redirect_principled_base_color_to_emission(source):
    """Expose imported glTF base color without Blender diffuse-bake drift.

    Pixel3D and other glTF producers connect an sRGB image to a Principled
    Base Color input.  Blender's selected-to-active DIFFUSE/COLOR bake can
    return almost-black values for these imported materials even though they
    render correctly.  Temporarily routing that exact Base Color socket to an
    Emission shader makes the transfer independent of lighting and BSDF
    interpretation.  The original graph is restored immediately after bake.
    """
    records = []
    configured_materials = set()
    for material in source.data.materials:
        if material is None or material in configured_materials:
            continue
        configured_materials.add(material)
        if not material.use_nodes or material.node_tree is None:
            raise RuntimeError(
                f"source material lacks a node graph: {material.name}"
            )
        tree = material.node_tree
        outputs = [
            node
            for node in tree.nodes
            if node.bl_idname == "ShaderNodeOutputMaterial"
            and node.is_active_output
        ]
        if len(outputs) != 1:
            raise RuntimeError(
                f"expected one active material output in {material.name}"
            )
        output = outputs[0]
        surface = output.inputs.get("Surface")
        original_links = list(surface.links) if surface is not None else []
        if len(original_links) != 1:
            raise RuntimeError(
                f"expected one surface link in source material {material.name}"
            )
        surface_shader = original_links[0].from_node
        if surface_shader.bl_idname != "ShaderNodeBsdfPrincipled":
            raise RuntimeError(
                "base-color emission bake requires a directly connected "
                f"Principled shader in {material.name}"
            )
        base_color = surface_shader.inputs.get("Base Color")
        if base_color is None:
            raise RuntimeError(
                f"Principled shader lacks Base Color in {material.name}"
            )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "AVEngine Temporary Base Color Bake"
        if base_color.is_linked:
            tree.links.new(base_color.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = base_color.default_value
        original_socket = original_links[0].from_socket
        tree.links.remove(original_links[0])
        tree.links.new(emission.outputs["Emission"], surface)
        records.append((tree, output, original_socket, emission))
    if not records:
        raise RuntimeError("source mesh has no usable Principled PBR material")
    return records


def restore_source_surface_shaders(records):
    for tree, output, original_socket, emission in records:
        surface = output.inputs["Surface"]
        for link in list(surface.links):
            tree.links.remove(link)
        tree.links.new(original_socket, surface)
        tree.nodes.remove(emission)


def select_bake_pair(source, proxy):
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy


def transfer_surface_attributes_bake(
    proxy,
    source,
    resolution,
    encoding_policy,
    base_color_gain,
):
    """Bake visible PBR color/roughness to a new watertight UV atlas."""
    print("WATERTIGHT_PROXY_STAGE surface_bake_unwrap_start", flush=True)
    smart_unwrap(proxy)
    print("WATERTIGHT_PROXY_STAGE surface_bake_unwrap_done", flush=True)

    proxy.data.materials.clear()
    material = bpy.data.materials.new(name="Watertight_Baked_PBR")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    output_node = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    principled = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output_node.inputs["Surface"],
    )
    proxy.data.materials.append(material)

    base_image = new_bake_image("Watertight_BaseColor", resolution, "sRGB")
    base_texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    base_texture.name = "Baked Base Color"
    base_texture.image = base_image
    material.node_tree.nodes.active = base_texture

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    diagonal = mesh_diagonal(proxy)
    ray_distance = max(diagonal * 0.03, 1e-6)
    cage_extrusion = max(diagonal * 0.005, 1e-6)
    select_bake_pair(source, proxy)
    print(
        "WATERTIGHT_PROXY_STAGE surface_bake_base_color_start "
        f"resolution={resolution}",
        flush=True,
    )
    emission_records = redirect_principled_base_color_to_emission(source)
    try:
        bpy.ops.object.bake(
            type="EMIT",
            use_selected_to_active=True,
            use_clear=True,
            margin=16,
            cage_extrusion=cage_extrusion,
            max_ray_distance=ray_distance,
        )
    finally:
        restore_source_surface_shaders(emission_records)
    if encoding_policy == "srgb-to-linear":
        reinterpret_baked_srgb_values_as_scene_linear(base_image)
    apply_base_color_gain(base_image, base_color_gain)
    base_image.pack()
    print("WATERTIGHT_PROXY_STAGE surface_bake_base_color_done", flush=True)

    roughness_image = new_bake_image(
        "Watertight_Roughness",
        resolution,
        "Non-Color",
    )
    roughness_texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    roughness_texture.name = "Baked Roughness"
    roughness_texture.image = roughness_image
    material.node_tree.nodes.active = roughness_texture
    select_bake_pair(source, proxy)
    print("WATERTIGHT_PROXY_STAGE surface_bake_roughness_start", flush=True)
    bpy.ops.object.bake(
        type="ROUGHNESS",
        use_selected_to_active=True,
        use_clear=True,
        margin=16,
        cage_extrusion=cage_extrusion,
        max_ray_distance=ray_distance,
    )
    roughness_image.pack()
    # Link target textures only after both selected-to-active bakes.  Linking an
    # image while writing that same image creates a Cycles circular dependency
    # and can mix the target's blank surface into the result.
    material.node_tree.links.new(
        base_texture.outputs["Color"],
        principled.inputs["Base Color"],
    )
    material.node_tree.links.new(
        roughness_texture.outputs["Color"],
        principled.inputs["Roughness"],
    )
    principled.inputs["Metallic"].default_value = 0.0
    print("WATERTIGHT_PROXY_STAGE surface_bake_roughness_done", flush=True)

    for polygon in proxy.data.polygons:
        polygon.use_smooth = True
    return {
        "backend": "bake",
        "bake_resolution": resolution,
        "bake_device": "CPU",
        "ray_distance": ray_distance,
        "cage_extrusion": cage_extrusion,
        "uv_layers": [layer.name for layer in proxy.data.uv_layers],
        "baked_images": [base_image.name, roughness_image.name],
        "base_color_bake_type": "EMIT_FROM_PRINCIPLED_BASE_COLOR",
        "color_attributes": [],
        "material_slots": [material.name],
        "metallic_policy": "constant_zero_for_nonmetallic_animal_surface",
        "base_color_encoding_policy": encoding_policy,
        "base_color_gain": list(base_color_gain),
    }


def transfer_surface_attributes(
    proxy,
    source,
    backend,
    bake_resolution,
    encoding_policy,
    base_color_gain,
):
    if backend == "bake":
        return transfer_surface_attributes_bake(
            proxy,
            source,
            bake_resolution,
            encoding_policy,
            base_color_gain,
        )
    if backend == "bvh":
        return transfer_surface_attributes_bvh(proxy, source)
    return transfer_surface_attributes_modifier(proxy, source)


def export_static(proxy, output):
    activate(proxy)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )


def main():
    args = parse_argv()
    source_path = require_input(args.source)
    attribute_source_path = (
        require_input(args.attribute_source) if args.attribute_source else source_path
    )
    output = require_output(args.output, "output GLB")
    manifest = require_output(args.manifest, "manifest")
    if not 96 <= args.voxel_resolution <= 512:
        raise SystemExit("--voxel-resolution must be in [96, 512]")
    if not 10000 <= args.target_faces <= 1000000:
        raise SystemExit("--target-faces must be in [10000, 1000000]")
    if not 0 <= args.smooth_iterations <= 8:
        raise SystemExit("--smooth-iterations must be in [0, 8]")
    if not 0.0 <= args.shrinkwrap_strength <= 1.0:
        raise SystemExit("--shrinkwrap-strength must be in [0, 1]")
    if not 0 <= args.post_shrinkwrap_smooth_iterations <= 8:
        raise SystemExit(
            "--post-shrinkwrap-smooth-iterations must be in [0, 8]"
        )
    if not 0 <= args.torso_fold_repair_iterations <= 20:
        raise SystemExit("--torso-fold-repair-iterations must be in [0, 20]")
    if args.bake_resolution not in (512, 1024, 2048, 4096):
        raise SystemExit("--bake-resolution must be 512, 1024, 2048, or 4096")
    if any(value <= 0.0 or value > 2.0 for value in args.base_color_gain):
        raise SystemExit("--base-color-gain values must be in (0, 2]")

    timer = stage("import_start", source=source_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source_path))
    meshes = real_meshes()
    if len(meshes) != 1:
        raise RuntimeError(f"expected one source mesh, got {[item.name for item in meshes]}")
    source = meshes[0]
    apply_transforms(source)
    source.name = "PBR_Surface_Authority"
    source_topology = topology_stats(source)
    diagonal = mesh_diagonal(source)
    voxel_size = diagonal / float(args.voxel_resolution)
    timer = stage(
        "import_done",
        timer,
        source_faces=source_topology["faces"],
        voxel_size=f"{voxel_size:.8f}",
    )

    attribute_source = source
    if attribute_source_path != source_path:
        existing_objects = set(bpy.context.scene.objects)
        timer = stage(
            "attribute_import_start",
            timer,
            source=attribute_source_path,
        )
        bpy.ops.import_scene.gltf(filepath=str(attribute_source_path))
        imported_attribute_meshes = [
            item
            for item in real_meshes()
            if item not in existing_objects
        ]
        if len(imported_attribute_meshes) != 1:
            raise RuntimeError(
                "expected one attribute-source mesh, got "
                f"{[item.name for item in imported_attribute_meshes]}"
            )
        attribute_source = imported_attribute_meshes[0]
        apply_transforms(attribute_source)
        attribute_source.name = "PBR_Attribute_Transfer_Authority"
        timer = stage(
            "attribute_import_done",
            timer,
            faces=len(attribute_source.data.polygons),
        )

    proxy = source.copy()
    proxy.data = source.data.copy()
    bpy.context.collection.objects.link(proxy)
    proxy.name = "Watertight_Runtime_Proxy"
    timer = stage("voxel_remesh_start", timer, resolution=args.voxel_resolution)
    voxel_remesh(proxy, voxel_size, args.smooth_iterations)
    voxel_topology = topology_stats(proxy)
    timer = stage("voxel_remesh_done", timer, faces=voxel_topology["faces"])
    timer = stage("shrinkwrap_start", timer)
    shrinkwrap_to_source(proxy, source, args.shrinkwrap_strength)
    timer = stage("shrinkwrap_done", timer)
    timer = stage(
        "post_shrinkwrap_smooth_start",
        timer,
        iterations=args.post_shrinkwrap_smooth_iterations,
    )
    post_shrinkwrap_smooth(proxy, args.post_shrinkwrap_smooth_iterations)
    timer = stage("post_shrinkwrap_smooth_done", timer)
    timer = stage(
        "torso_fold_repair_start",
        timer,
        iterations=args.torso_fold_repair_iterations,
    )
    torso_fold_repair = repair_normalized_torso_folds(
        proxy,
        args.torso_fold_repair_iterations,
    )
    timer = stage(
        "torso_fold_repair_done",
        timer,
        selected_vertices=torso_fold_repair["selected_vertices"],
    )
    timer = stage("decimate_start", timer, target_faces=args.target_faces)
    actual_faces = decimate(proxy, args.target_faces)
    timer = stage("decimate_done", timer, faces=actual_faces)
    timer = stage(
        "surface_transfer_start",
        timer,
        authority_faces=len(attribute_source.data.polygons),
    )
    attributes = transfer_surface_attributes(
        proxy,
        attribute_source,
        args.attribute_transfer_backend,
        args.bake_resolution,
        args.base_color_encoding_policy,
        tuple(args.base_color_gain),
    )
    timer = stage("surface_transfer_done", timer)
    if args.double_sided:
        for material in proxy.data.materials:
            if material is not None:
                material.use_backface_culling = False
    final_topology = topology_stats(proxy)
    if final_topology["boundary_edges"] or final_topology["wire_edges"]:
        raise RuntimeError(f"watertight proxy has open topology: {final_topology}")
    if final_topology["nonmanifold_edges_over_two_faces"]:
        raise RuntimeError(f"watertight proxy is non-manifold: {final_topology}")

    source.hide_render = True
    source.hide_viewport = True
    if attribute_source is not source:
        attribute_source.hide_render = True
        attribute_source.hide_viewport = True
    timer = stage("export_start", timer)
    export_static(proxy, output)
    stage("export_done", timer, output=output)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        },
        "attribute_input": {
            "path": str(attribute_source_path),
            "sha256": sha256_file(attribute_source_path),
            "size_bytes": attribute_source_path.stat().st_size,
            "same_as_geometry_input": attribute_source_path == source_path,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "parameters": {
            "voxel_resolution": args.voxel_resolution,
            "voxel_size": voxel_size,
            "target_faces": args.target_faces,
            "smooth_iterations": args.smooth_iterations,
            "shrinkwrap_strength": args.shrinkwrap_strength,
            "post_shrinkwrap_smooth_iterations": (
                args.post_shrinkwrap_smooth_iterations
            ),
            "torso_fold_repair_iterations": (
                args.torso_fold_repair_iterations
            ),
            "double_sided": bool(args.double_sided),
            "attribute_transfer_backend": args.attribute_transfer_backend,
            "bake_resolution": args.bake_resolution,
            "base_color_encoding_policy": args.base_color_encoding_policy,
            "base_color_gain": list(args.base_color_gain),
        },
        "topology": {
            "source": source_topology,
            "after_voxel_remesh": voxel_topology,
            "final": final_topology,
        },
        "surface_attributes": attributes,
        "torso_fold_repair": torso_fold_repair,
        "authority_contract": {
            "attribute_source_pbr_material_reused": (
                args.attribute_transfer_backend != "bake"
            ),
            "attribute_source_uvs_transferred_by_nearest_surface": (
                args.attribute_transfer_backend in {"bvh", "data-transfer"}
            ),
            "attribute_source_pbr_baked_to_new_uv_atlas": (
                args.attribute_transfer_backend == "bake"
            ),
            "full_resolution_source_remains_geometry_authority": True,
            "source_geometry_replaced": True,
            "approved_skeleton_or_animation_touched": False,
        },
        "actual_faces": actual_faces,
        "status": "research_candidate_pending_static_and_animation_qa",
        "formal_dataset_registration_authorized": False,
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "WATERTIGHT_TEXTURED_PROXY_OK "
        f"faces={actual_faces} voxel_size={voxel_size:.8f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
