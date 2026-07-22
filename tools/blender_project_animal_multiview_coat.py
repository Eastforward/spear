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
from collections import deque
import copy
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
import numpy as np


SCHEMA = "avengine_generated_animal_multiview_coat_projection_v2"
VIEW_ORDER = ("front", "back", "left", "right")


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--source-view-dir", type=Path, required=True)
    parser.add_argument("--edited-view-dir", type=Path, required=True)
    parser.add_argument(
        "--edited-mask-dir",
        type=Path,
        help=(
            "Optional front/back/left/right grayscale animal masks. When "
            "present, mask alpha is the foreground authority and the colour "
            "background heuristic is not used."
        ),
    )
    parser.add_argument(
        "--neutral-shading-view-dir",
        type=Path,
        help=(
            "Optional neutral-grey same-camera render views used by "
            "neutral_shading_division to remove preview illumination."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="animal_coat",
        help="Portable lowercase identifier used in output texture and GLB filenames.",
    )
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--visibility-tolerance-ratio", type=float, default=0.003)
    parser.add_argument("--minimum-direct-coverage", type=float, default=0.60)
    parser.add_argument(
        "--view-fusion-mode",
        choices=("weighted_average", "dominant_facing_view"),
        default="weighted_average",
        help=(
            "Fuse compatible views by weighted average, or keep only the "
            "most front-facing valid view per surface vertex when discrete "
            "breed markings must not be averaged into grey seams."
        ),
    )
    parser.add_argument(
        "--edited-foreground-distance-threshold",
        type=float,
        default=0.025,
        help=(
            "Minimum scene-linear RGB distance from the fitted edited-view "
            "background. Samples that still contain FLUX background are not "
            "allowed to become coat texels."
        ),
    )
    parser.add_argument(
        "--edited-foreground-chroma-threshold",
        type=float,
        default=0.020,
        help="Minimum scene-linear chromatic contrast from a neutral background.",
    )
    parser.add_argument(
        "--edited-foreground-luminance-threshold",
        type=float,
        default=0.050,
        help="Minimum absolute Rec.709 luminance contrast for neutral black/white fur.",
    )
    parser.add_argument("--luminance-transfer-strength", type=float, default=0.15)
    parser.add_argument(
        "--colour-transfer-mode",
        choices=(
            "relative_chroma",
            "relative_rgb",
            "neutral_shading_division",
            "absolute_edited_chroma",
            "absolute_edited_rgb",
        ),
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
    parser.add_argument(
        "--absolute-rgb-strength",
        type=float,
        default=0.85,
        help=(
            "Blend strength in [0,1] for absolute_edited_rgb. This transfers "
            "the spatial FLUX-edited RGB field, not one global RGB factor."
        ),
    )
    parser.add_argument(
        "--relative-rgb-strength",
        type=float,
        default=1.0,
        help=(
            "Strength in [0,1] for the spatial edited/source RGB reflectance "
            "ratio. Unlike absolute RGB, common review lighting cancels."
        ),
    )
    parser.add_argument(
        "--relative-rgb-epsilon",
        type=float,
        default=0.005,
        help="Scene-linear stabilizer in [0.001,0.025] for dark source texels.",
    )
    parser.add_argument(
        "--pattern-luminance-strength",
        type=float,
        default=0.0,
        help=(
            "Gain in [0,2] for the edited/source spatial log-luminance delta. "
            "Keep 0 for colour-only coats, 1 follows the measured edit, and a "
            "reviewed value above 1 may recover pattern contrast lost by a very "
            "dark source texture."
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


def fitted_background_rgb(image, xy):
    """Fit the nearly planar review background from a narrow image border."""
    height, width = image.shape[:2]
    border = max(4, min(height, width) // 64)
    yy, xx = np.mgrid[0:height, 0:width]
    mask = (xx < border) | (xx >= width - border) | (yy < border) | (yy >= height - border)
    design = np.column_stack(
        (
            xx[mask] / max(width - 1, 1),
            yy[mask] / max(height - 1, 1),
            np.ones(int(np.count_nonzero(mask))),
        )
    )
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design, image[mask, :3], rcond=None
    )
    query = np.column_stack(
        (
            xy[:, 0] / max(width - 1, 1),
            xy[:, 1] / max(height - 1, 1),
            np.ones(len(xy)),
        )
    )
    return np.clip(query @ coefficients, 0.0, 1.0)


def edited_foreground_mask(
    image,
    xy,
    sampled_rgb,
    minimum_distance,
    minimum_chroma,
    minimum_luminance,
):
    """Reject edit pixels whose colour is still the generated background."""
    background = fitted_background_rgb(image, xy)
    delta = sampled_rgb - background
    distance = np.linalg.norm(delta, axis=1)
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    luminance_delta = np.abs(delta @ rec709)
    neutral_removed = delta - delta.mean(axis=1, keepdims=True)
    chroma_delta = np.linalg.norm(neutral_removed, axis=1)
    foreground = (distance >= minimum_distance) & (
        (chroma_delta >= minimum_chroma)
        | (luminance_delta >= minimum_luminance)
    )
    return foreground, distance, chroma_delta, luminance_delta


def topology_fill(values, covered, edges, vertices):
    """Fill through mesh edges and coincident export-seam vertices only."""
    if not np.any(covered):
        raise RuntimeError("cannot topology-fill a field without covered vertices")
    adjacency = [[] for _ in range(len(covered))]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    weld_tolerance = max(diagonal * 1.0e-7, 1.0e-9)
    coincident = {}
    for index, point in enumerate(vertices):
        key = tuple(np.rint(point / weld_tolerance).astype(np.int64))
        coincident.setdefault(key, []).append(index)
    for indices in coincident.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for index in indices[1:]:
            adjacency[anchor].append(index)
            adjacency[index].append(anchor)
    owner = np.full(len(covered), -1, dtype=np.int64)
    queue = deque()
    for index in np.flatnonzero(covered):
        owner[index] = index
        queue.append(int(index))
    while queue:
        index = queue.popleft()
        for neighbour in adjacency[index]:
            if owner[neighbour] >= 0:
                continue
            owner[neighbour] = owner[index]
            queue.append(neighbour)
    missing = np.flatnonzero(~covered)
    fillable = owner >= 0
    fillable_missing = missing[fillable[missing]]
    values[fillable_missing] = values[owner[fillable_missing]]
    return values, fillable


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
    edited_foreground_distance_threshold,
    edited_foreground_chroma_threshold,
    edited_foreground_luminance_threshold,
    relative_rgb_epsilon,
    view_fusion_mode,
    edited_mask_dir,
    neutral_shading_view_dir,
):
    bvh = BVHTree.FromPolygons(
        [tuple(item) for item in vertices],
        [tuple(poly.vertices) for poly in mesh.data.polygons],
        all_triangles=False,
    )
    accumulation = np.zeros((len(vertices), 3), dtype=np.float64)
    edited_rgb_accumulation = np.zeros((len(vertices), 3), dtype=np.float64)
    shading_rgb_accumulation = np.zeros((len(vertices), 3), dtype=np.float64)
    luminance_accumulation = np.zeros(len(vertices), dtype=np.float64)
    weights = np.zeros(len(vertices), dtype=np.float64)
    per_view = {}
    cameras = []
    scene = bpy.context.scene

    for name in VIEW_ORDER:
        source_handle, source = image_array(source_dir / f"{name}.png")
        edited_handle, edited = image_array(edited_dir / f"{name}.png")
        mask_handle = None
        edited_mask = None
        shading_handle = None
        shading_rgb_image = None
        if edited_mask_dir is not None:
            mask_handle, edited_mask = image_array(
                edited_mask_dir / f"{name}.png"
            )
        if neutral_shading_view_dir is not None:
            shading_handle, shading_rgb_image = image_array(
                neutral_shading_view_dir / f"{name}.png"
            )
        if source.shape != edited.shape or source.shape[:2] != (512, 512):
            raise RuntimeError(f"source/edited view mismatch for {name}")
        if edited_mask is not None and edited_mask.shape[:2] != (512, 512):
            raise RuntimeError(f"edited mask canvas mismatch for {name}")
        if shading_rgb_image is not None and shading_rgb_image.shape[:2] != (512, 512):
            raise RuntimeError(f"neutral shading canvas mismatch for {name}")
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
        shading_rgb = (
            np.clip(bilinear(shading_rgb_image, xy), 0.0, 1.0)
            if shading_rgb_image is not None
            else np.ones_like(edited_rgb)
        )
        if edited_mask is not None:
            mask_probability = bilinear(edited_mask, xy)[:, 0]
            foreground = mask_probability >= 0.5
            foreground_distance = np.ones(len(foreground), dtype=np.float64)
            foreground_chroma = np.ones(len(foreground), dtype=np.float64)
            foreground_luminance = np.ones(len(foreground), dtype=np.float64)
        else:
            (
                foreground,
                foreground_distance,
                foreground_chroma,
                foreground_luminance,
            ) = edited_foreground_mask(
                edited,
                xy,
                edited_rgb,
                edited_foreground_distance_threshold,
                edited_foreground_chroma_threshold,
                edited_foreground_luminance_threshold,
            )
        rejected_background_count = int(np.count_nonzero(~foreground))
        visible = visible[foreground]
        source_rgb = source_rgb[foreground]
        edited_rgb = edited_rgb[foreground]
        shading_rgb = shading_rgb[foreground]
        facing_visible = facing[visible]
        if not len(visible):
            raise RuntimeError(
                f"edited view contains no foreground samples for visible {name} vertices"
            )
        source_log = np.log(source_rgb + 0.025)
        edited_log = np.log(edited_rgb + 0.025)
        source_luminance = source_log.mean(axis=1, keepdims=True)
        edited_luminance = edited_log.mean(axis=1, keepdims=True)
        luminance_delta = np.clip(
            edited_luminance[:, 0] - source_luminance[:, 0],
            math.log(0.25),
            math.log(4.0),
        )
        source_chroma = source_log - source_luminance
        edited_chroma = edited_log - edited_luminance
        if colour_transfer_mode == "relative_rgb":
            colour_field = np.clip(
                np.log(edited_rgb + relative_rgb_epsilon)
                - np.log(source_rgb + relative_rgb_epsilon),
                math.log(0.02),
                math.log(50.0),
            )
        elif colour_transfer_mode == "relative_chroma":
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
        view_weight = np.square(np.clip(facing_visible, 0.0, 1.0))
        if view_fusion_mode == "dominant_facing_view":
            better = view_weight > weights[visible]
            chosen = visible[better]
            chosen_weight = view_weight[better]
            accumulation[chosen] = colour_field[better] * chosen_weight[:, None]
            edited_rgb_accumulation[chosen] = (
                edited_rgb[better] * chosen_weight[:, None]
            )
            shading_rgb_accumulation[chosen] = (
                shading_rgb[better] * chosen_weight[:, None]
            )
            luminance_accumulation[chosen] = luminance_delta[better] * chosen_weight
            weights[chosen] = chosen_weight
        else:
            accumulation[visible] += colour_field * view_weight[:, None]
            edited_rgb_accumulation[visible] += edited_rgb * view_weight[:, None]
            shading_rgb_accumulation[visible] += shading_rgb * view_weight[:, None]
            luminance_accumulation[visible] += luminance_delta * view_weight
            weights[visible] += view_weight
        per_view[name] = {
            "candidate_vertex_count": int(len(candidates)),
            "visible_vertex_count": int(len(visible)),
            "edited_background_rejected_vertex_count": rejected_background_count,
            "foreground_authority": (
                "external_alpha_mask" if edited_mask is not None else "colour_heuristic"
            ),
            "minimum_accepted_foreground_distance": float(
                foreground_distance[foreground].min()
            ),
            "minimum_accepted_foreground_chroma": float(
                foreground_chroma[foreground].min()
            ),
            "minimum_accepted_foreground_luminance_delta": float(
                foreground_luminance[foreground].min()
            ),
            "mean_facing_weight": float(view_weight.mean()),
        }
        bpy.data.images.remove(source_handle)
        bpy.data.images.remove(edited_handle)
        if mask_handle is not None:
            bpy.data.images.remove(mask_handle)
        if shading_handle is not None:
            bpy.data.images.remove(shading_handle)

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
    edited_rgb_field = np.zeros_like(edited_rgb_accumulation)
    edited_rgb_field[covered] = (
        edited_rgb_accumulation[covered] / weights[covered, None]
    )
    shading_rgb_field = np.zeros_like(shading_rgb_accumulation)
    shading_rgb_field[covered] = (
        shading_rgb_accumulation[covered] / weights[covered, None]
    )
    luminance_field = np.zeros(len(vertices), dtype=np.float64)
    luminance_field[covered] = luminance_accumulation[covered] / weights[covered]

    edges = [tuple(edge.vertices) for edge in mesh.data.edges]
    colour_field, field_available = topology_fill(
        colour_field, covered, edges, vertices
    )
    edited_rgb_field, rgb_available = topology_fill(
        edited_rgb_field, covered, edges, vertices
    )
    shading_rgb_field, shading_available = topology_fill(
        shading_rgb_field, covered, edges, vertices
    )
    luminance_field, luminance_available = topology_fill(
        luminance_field, covered, edges, vertices
    )
    if (
        not np.array_equal(field_available, rgb_available)
        or not np.array_equal(field_available, shading_available)
        or not np.array_equal(field_available, luminance_available)
    ):
        raise RuntimeError("coat field fills disagree on available mesh components")
    return (
        colour_field.astype(np.float32),
        edited_rgb_field.astype(np.float32),
        shading_rgb_field.astype(np.float32),
        luminance_field.astype(np.float32),
        covered,
        field_available,
        per_view,
        cameras,
    )


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


def decode_glb(path: Path):
    raw = path.read_bytes()
    if len(raw) < 28:
        raise RuntimeError("GLB is truncated")
    magic, version, declared = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared != len(raw):
        raise RuntimeError("GLB header is invalid")
    offset = 12
    chunks = {}
    while offset < len(raw):
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[chunk_type] = raw[offset : offset + length]
        offset += length
    if set(chunks) != {0x4E4F534A, 0x004E4942}:
        raise RuntimeError("GLB must contain exactly JSON and BIN chunks")
    document = json.loads(chunks[0x4E4F534A].decode("utf-8").rstrip(" \x00"))
    declared_binary = document["buffers"][0]["byteLength"]
    return document, chunks[0x004E4942][:declared_binary]


def buffer_view_payload(document, binary, index):
    view = document["bufferViews"][index]
    start = int(view.get("byteOffset", 0))
    length = int(view["byteLength"])
    payload = binary[start : start + length]
    if len(payload) != length:
        raise RuntimeError(f"bufferView {index} exceeds embedded BIN")
    return payload


def encode_glb(document, binary):
    value = copy.deepcopy(document)
    value["buffers"][0]["byteLength"] = len(binary)
    encoded_json = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded_json += b" " * ((-len(encoded_json)) % 4)
    encoded_binary = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded_json) + 8 + len(encoded_binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(encoded_json), 0x4E4F534A),
            encoded_json,
            struct.pack("<II", len(encoded_binary), 0x004E4942),
            encoded_binary,
        )
    )


def patch_embedded_base_color(source_glb, texture_png, output_glb):
    document, binary = decode_glb(source_glb)
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    if len(materials) != 1:
        raise RuntimeError("container patch requires exactly one source material")
    texture_index = materials[0].get("pbrMetallicRoughness", {}).get(
        "baseColorTexture", {}
    ).get("index")
    if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
        raise RuntimeError("source GLB has no unambiguous Base Color texture")
    image_index = textures[texture_index].get("source")
    if not isinstance(image_index, int) or not 0 <= image_index < len(images):
        raise RuntimeError("source GLB has no unambiguous Base Color image")
    image = images[image_index]
    view_index = image.get("bufferView")
    if not isinstance(view_index, int) or image.get("mimeType") != "image/png":
        raise RuntimeError("source Base Color must be one embedded PNG")
    payload = texture_png.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("baked Base Color is not a PNG")

    updated = copy.deepcopy(document)
    target = updated["bufferViews"][view_index]
    start = int(target.get("byteOffset", 0))
    old_length = int(target["byteLength"])
    old_end = start + old_length
    later_offsets = [
        int(view.get("byteOffset", 0))
        for index, view in enumerate(updated["bufferViews"])
        if index != view_index and int(view.get("byteOffset", 0)) >= old_end
    ]
    tail_start = min(later_offsets, default=len(binary))
    padded = payload + b"\x00" * ((-len(payload)) % 4)
    new_binary = binary[:start] + padded + binary[tail_start:]
    delta = start + len(padded) - tail_start
    target["byteLength"] = len(payload)
    for index, view in enumerate(updated["bufferViews"]):
        if index != view_index and int(view.get("byteOffset", 0)) >= tail_start:
            view["byteOffset"] = int(view.get("byteOffset", 0)) + delta
    updated["buffers"][0]["byteLength"] = len(new_binary)

    unchanged = 0
    for index in range(len(document["bufferViews"])):
        if index == view_index:
            continue
        if buffer_view_payload(document, binary, index) != buffer_view_payload(
            updated, new_binary, index
        ):
            raise RuntimeError(f"container patch changed protected bufferView {index}")
        unchanged += 1
    output_glb.write_bytes(encode_glb(updated, new_binary))
    decoded, decoded_binary = decode_glb(output_glb)
    if buffer_view_payload(decoded, decoded_binary, view_index) != payload:
        raise RuntimeError("Base Color payload failed GLB readback")
    for key in ("nodes", "meshes", "skins", "accessors", "animations"):
        if decoded.get(key) != document.get(key):
            raise RuntimeError(f"container patch changed protected GLB JSON: {key}")
    return {
        "method": "embedded_base_color_buffer_view_replacement_v1",
        "base_color_image_index": image_index,
        "base_color_buffer_view_index": view_index,
        "source_payload_size_bytes": old_length,
        "replacement_payload_size_bytes": len(payload),
        "non_target_buffer_views_unchanged": unchanged,
        "protected_json_sections_unchanged": [
            "nodes",
            "meshes",
            "skins",
            "accessors",
            "animations",
        ],
    }


def main():
    args = parse_argv()
    if args.texture_size not in {512, 1024, 2048, 4096}:
        raise RuntimeError("--texture-size must be 512, 1024, 2048, or 4096")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_stem) is None:
        raise RuntimeError("--output-stem must match [a-z0-9][a-z0-9_-]{0,63}")
    if not 0.40 <= args.minimum_direct_coverage <= 0.95:
        raise RuntimeError("--minimum-direct-coverage must be in [0.40, 0.95]")
    if not 0.005 <= args.edited_foreground_distance_threshold <= 0.20:
        raise RuntimeError(
            "--edited-foreground-distance-threshold must be in [0.005, 0.20]"
        )
    if not 0.0 <= args.edited_foreground_chroma_threshold <= 0.20:
        raise RuntimeError(
            "--edited-foreground-chroma-threshold must be in [0, 0.20]"
        )
    if not 0.0 <= args.edited_foreground_luminance_threshold <= 0.40:
        raise RuntimeError(
            "--edited-foreground-luminance-threshold must be in [0, 0.40]"
        )
    if not 0.0 <= args.luminance_transfer_strength <= 1.0:
        raise RuntimeError("--luminance-transfer-strength must be in [0, 1]")
    if not 0.0 <= args.absolute_chroma_strength <= 1.0:
        raise RuntimeError("--absolute-chroma-strength must be in [0, 1]")
    if not 0.0 <= args.absolute_rgb_strength <= 1.0:
        raise RuntimeError("--absolute-rgb-strength must be in [0, 1]")
    if not 0.0 <= args.relative_rgb_strength <= 1.0:
        raise RuntimeError("--relative-rgb-strength must be in [0, 1]")
    if not 0.001 <= args.relative_rgb_epsilon <= 0.025:
        raise RuntimeError("--relative-rgb-epsilon must be in [0.001, 0.025]")
    if not 0.0 <= args.pattern_luminance_strength <= 2.0:
        raise RuntimeError("--pattern-luminance-strength must be in [0, 2]")
    input_glb = args.input_glb.resolve()
    source_dir = args.source_view_dir.resolve()
    edited_dir = args.edited_view_dir.resolve()
    edited_mask_dir = args.edited_mask_dir.resolve() if args.edited_mask_dir else None
    neutral_shading_view_dir = (
        args.neutral_shading_view_dir.resolve()
        if args.neutral_shading_view_dir
        else None
    )
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
    if edited_mask_dir is not None:
        for name in VIEW_ORDER:
            path = edited_mask_dir / f"{name}.png"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"missing edited foreground mask: {path}")
    if args.colour_transfer_mode == "neutral_shading_division" and (
        neutral_shading_view_dir is None
    ):
        raise RuntimeError(
            "neutral_shading_division requires --neutral-shading-view-dir"
        )
    if neutral_shading_view_dir is not None:
        for name in VIEW_ORDER:
            path = neutral_shading_view_dir / f"{name}.png"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"missing neutral shading view: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    mesh = primary_skinned_mesh()
    for armature in [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]:
        armature.data.pose_position = "REST"
    for other in [item for item in bpy.context.scene.objects if item.type == "MESH"]:
        other.hide_render = other != mesh
    bpy.context.view_layer.update()

    _material, _principled, _base_nodes, original_base_image = base_colour_binding(mesh)
    vertices, normals, polygons = world_geometry(mesh)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tolerance = diagonal * args.visibility_tolerance_ratio
    (
        colour_field,
        edited_rgb_field,
        shading_rgb_field,
        luminance_field,
        covered,
        field_available,
        per_view,
        cameras,
    ) = vertex_log_ratios(
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
        args.edited_foreground_distance_threshold,
        args.edited_foreground_chroma_threshold,
        args.edited_foreground_luminance_threshold,
        args.relative_rgb_epsilon,
        args.view_fusion_mode,
        edited_mask_dir,
        neutral_shading_view_dir,
    )

    loop_vertex = np.empty(len(mesh.data.loops), dtype=np.int32)
    mesh.data.loops.foreach_get("vertex_index", loop_vertex)
    base_rgb = sample_base_texture_for_loops(mesh, original_base_image)
    if args.colour_transfer_mode == "neutral_shading_division":
        rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        neutral_base_colour = 0.5
        shading_luminance = shading_rgb_field[loop_vertex] @ rec709
        illumination = np.maximum(
            shading_luminance / neutral_base_colour,
            0.08,
        )
        target_rgb = np.clip(
            edited_rgb_field[loop_vertex] / illumination[:, None],
            0.0,
            1.0,
        )
        target_rgb = np.where(
            field_available[loop_vertex, None], target_rgb, base_rgb
        )
        corner_rgb = np.clip(
            (1.0 - args.absolute_rgb_strength) * base_rgb
            + args.absolute_rgb_strength * target_rgb,
            0.0,
            1.0,
        )
    elif args.colour_transfer_mode == "absolute_edited_rgb":
        target_rgb = np.where(
            field_available[loop_vertex, None],
            edited_rgb_field[loop_vertex],
            base_rgb,
        )
        corner_rgb = np.clip(
            (1.0 - args.absolute_rgb_strength) * base_rgb
            + args.absolute_rgb_strength * target_rgb,
            0.0,
            1.0,
        )
    elif args.colour_transfer_mode == "relative_rgb":
        ratio = np.exp(args.relative_rgb_strength * colour_field[loop_vertex])
        corner_rgb = np.clip(
            (base_rgb + args.relative_rgb_epsilon) * ratio
            - args.relative_rgb_epsilon,
            0.0,
            1.0,
        )
    elif args.colour_transfer_mode == "relative_chroma":
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
    if args.colour_transfer_mode not in {
        "absolute_edited_rgb",
        "relative_rgb",
        "neutral_shading_division",
    }:
        rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        base_luminance = base_rgb @ rec709
        candidate_luminance = candidate_rgb @ rec709
        desired_luminance = np.clip(
            base_luminance
            * np.exp(args.pattern_luminance_strength * luminance_field[loop_vertex]),
            0.0,
            1.0,
        )
        luminance_normalization = (desired_luminance + 1.0e-5) / (
            candidate_luminance + 1.0e-5
        )
        corner_rgb = np.clip(
            candidate_rgb * luminance_normalization[:, None], 0.0, 1.0
        )
    texture_path = output_root / f"{args.output_stem}_base_color.png"
    bake_corner_colours(mesh, corner_rgb, texture_path, args.texture_size)

    for camera in cameras:
        bpy.data.objects.remove(camera, do_unlink=True)

    output_glb = output_root / f"animated_walk_idle_{args.output_stem}.glb"
    container_patch = patch_embedded_base_color(
        input_glb, texture_path, output_glb
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
        "edited_mask_dir": str(edited_mask_dir) if edited_mask_dir else None,
        "neutral_shading_view_dir": (
            str(neutral_shading_view_dir) if neutral_shading_view_dir else None
        ),
        "neutral_shading_division": {
            "enabled": args.colour_transfer_mode == "neutral_shading_division",
            "neutral_base_color_linear": 0.5,
            "minimum_illumination_factor": 0.08,
            "method": "edited_rgb_divided_by_neutral_same_camera_rec709_shading_v1",
        },
        "foreground_authority": (
            "external_alpha_mask" if edited_mask_dir else "colour_heuristic"
        ),
        "projection_method": (
            "geometry_locked_multiview_surface_field_with_optional_neutral_"
            "shading_division_v6"
        ),
        "not_global_rgb_factor": True,
        "colour_transfer_mode": args.colour_transfer_mode,
        "absolute_chroma_strength": args.absolute_chroma_strength,
        "absolute_rgb_strength": args.absolute_rgb_strength,
        "relative_rgb_strength": args.relative_rgb_strength,
        "relative_rgb_epsilon": args.relative_rgb_epsilon,
        "relative_rgb_is_spatial_edited_over_source_reflectance_ratio": (
            args.colour_transfer_mode == "relative_rgb"
        ),
        "absolute_rgb_is_spatial_flux_field_not_global_factor": (
            args.colour_transfer_mode == "absolute_edited_rgb"
        ),
        "pattern_luminance_strength": args.pattern_luminance_strength,
        "spatial_luminance_method": "edited_over_source_log_luminance_delta_v1",
        "explicit_srgb_to_linear_input_decode": True,
        "bake_output_colourspace": "sRGB",
        "luminance_transfer_strength": args.luminance_transfer_strength,
        "per_uv_corner_rec709_linear_luminance_preserved": (
            args.pattern_luminance_strength == 0.0
        ),
        "geometry_skin_skeleton_and_actions_preserved_by_design": True,
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "loop_count": len(mesh.data.loops),
        "directly_covered_vertex_count": int(np.count_nonzero(covered)),
        "nearest_filled_vertex_count": int(np.count_nonzero(~covered)),
        "field_available_vertex_count": int(np.count_nonzero(field_available)),
        "original_texture_preserved_vertex_count": int(
            np.count_nonzero(~field_available)
        ),
        "direct_coverage_ratio": float(np.mean(covered)),
        "minimum_direct_coverage_required": args.minimum_direct_coverage,
        "visibility_tolerance": tolerance,
        "edited_foreground_distance_threshold": (
            args.edited_foreground_distance_threshold
        ),
        "edited_foreground_chroma_threshold": (
            args.edited_foreground_chroma_threshold
        ),
        "edited_foreground_luminance_threshold": (
            args.edited_foreground_luminance_threshold
        ),
        "view_fusion_mode": args.view_fusion_mode,
        "uncovered_fill_method": (
            "nearest_covered_vertex_over_mesh_edges_and_coincident_export_seams_v2"
        ),
        "per_view": per_view,
        "texture_size": args.texture_size,
        "base_color_texture": str(texture_path),
        "output_glb": str(output_glb),
        "container_patch": container_patch,
        "readback": {
            "animations": animations,
            "skin_count": len(document.get("skins", [])),
            "skinned_primitive_count": len(skinned),
        },
        "material_policy": "all_source_material_fields_except_base_color_payload_preserved",
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
