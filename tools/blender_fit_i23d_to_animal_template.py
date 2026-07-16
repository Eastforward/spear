#!/usr/bin/env python3
"""Fit an image-to-3D animal guide onto a stable animated template.

The generated guide supplies breed silhouette and PBR appearance only.  The
output keeps the template topology, armature, skin weights, and Walk/Idle
actions, so colour/size instances do not need to be rigged independently.

Example::

  blender -b --python tools/blender_fit_i23d_to_animal_template.py -- \
    --template-glb /path/to/Dog.glb \
    --guide-glb /path/to/beagle_runtime_lod.glb \
    --guide-front-axis negative-y \
    --output-glb /path/to/beagle_template.glb \
    --manifest /path/to/beagle_template.manifest.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_robust_swap_mesh_keep_rig as robust  # noqa: E402
from tools.robust_skin_transfer import (  # noqa: E402
    REGION_NAMES,
    _closest_point_barycentric,
    coarse_region_labels,
    face_region_labels,
    mesh_bounds,
    target_region_labels_from_source_proximity,
)


SCHEMA = "avengine_stable_animal_breed_template_fit_v1"
CARDINAL_YAW_DEGREES = {
    "positive-x": 0.0,
    "positive-y": -90.0,
    "negative-x": 180.0,
    "negative-y": 90.0,
}


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-glb", type=Path, required=True)
    parser.add_argument("--guide-glb", type=Path, required=True)
    parser.add_argument(
        "--guide-front-axis",
        choices=tuple(CARDINAL_YAW_DEGREES),
        required=True,
    )
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subdivision-levels", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--fit-strength", type=float, default=0.70)
    parser.add_argument("--max-displacement-ratio", type=float, default=0.10)
    parser.add_argument(
        "--geometry-fit-mode",
        choices=("all", "axial-only"),
        default="all",
        help=(
            "Fit every semantic region (legacy behavior), or fit only the "
            "torso/head/tail while keeping all four template limbs and their "
            "native skinning geometry unchanged.  axial-only is the stable "
            "choice when a single-image guide has ambiguous far-side limbs."
        ),
    )
    parser.add_argument("--smooth-iterations", type=int, default=4)
    parser.add_argument("--smooth-blend", type=float, default=0.45)
    parser.add_argument("--foot-lock-height-ratio", type=float, default=0.055)
    parser.add_argument(
        "--appearance-transfer",
        choices=("vertex-color", "region-atlas", "bake", "projected-uv"),
        default="vertex-color",
        help=(
            "Sample aligned guide PBR into the subdivided stable template's "
            "vertex colors (default), rasterize semantic-region constrained "
            "surface samples to the template UV atlas, or use a raw "
            "selected-to-active bake. "
            "The legacy projected-uv mode is retained only for reproducibility."
        ),
    )
    parser.add_argument(
        "--bake-resolution",
        type=int,
        choices=(512, 1024, 2048, 4096),
        default=1024,
    )
    parser.add_argument(
        "--bake-max-ray-distance-ratio",
        type=float,
        default=0.12,
        help="Selected-to-active bake ray distance as template bbox diagonal ratio.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".glb", ".gltf"}:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_new_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def world_vertices(obj) -> np.ndarray:
    return np.asarray(
        [tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices],
        dtype=np.float64,
    )


def triangulated_faces(obj) -> np.ndarray:
    faces = []
    for polygon in obj.data.polygons:
        indices = list(map(int, polygon.vertices))
        for offset in range(1, len(indices) - 1):
            faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not faces:
        raise RuntimeError(f"mesh has no faces: {obj.name}")
    return np.asarray(faces, dtype=np.int64)


def identify_template():
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if not meshes or len(armatures) != 1:
        raise RuntimeError("template must contain a mesh and exactly one armature")
    mesh = max(meshes, key=lambda item: len(item.data.vertices))
    armature = armatures[0]
    if not any(modifier.type == "ARMATURE" for modifier in mesh.modifiers):
        raise RuntimeError("template mesh is not skinned")
    action_names = sorted(action.name for action in bpy.data.actions)
    if not any("walk" in name.lower() for name in action_names):
        raise RuntimeError("template has no Walking action")
    if not any("idle" in name.lower() for name in action_names):
        raise RuntimeError("template has no Idle action")
    for other in meshes:
        if other is not mesh:
            bpy.data.objects.remove(other, do_unlink=True)
    return mesh, armature, action_names


def apply_template_subdivision(mesh, levels: int):
    imported = {
        "vertices": len(mesh.data.vertices),
        "polygons": len(mesh.data.polygons),
        "weighted_vertices": sum(bool(vertex.groups) for vertex in mesh.data.vertices),
    }
    position_weld = robust.weld_target_position_duplicates(mesh)
    before = {
        "vertices": len(mesh.data.vertices),
        "polygons": len(mesh.data.polygons),
        "weighted_vertices": sum(bool(vertex.groups) for vertex in mesh.data.vertices),
    }
    if before["weighted_vertices"] != before["vertices"]:
        raise RuntimeError("position weld did not preserve complete skin weights")
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    modifier = mesh.modifiers.new(name="BreedTemplateSubdivision", type="SUBSURF")
    modifier.subdivision_type = "CATMULL_CLARK"
    modifier.levels = levels
    modifier.render_levels = levels
    bpy.ops.object.modifier_move_to_index(modifier=modifier.name, index=0)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    after = {
        "vertices": len(mesh.data.vertices),
        "polygons": len(mesh.data.polygons),
        "weighted_vertices": sum(bool(vertex.groups) for vertex in mesh.data.vertices),
    }
    if after["vertices"] <= before["vertices"] or after["weighted_vertices"] != after["vertices"]:
        raise RuntimeError("subdivision did not preserve interpolated skin weights")
    return {
        "levels": levels,
        "imported": imported,
        "position_weld": position_weld,
        "before": before,
        "after": after,
    }


def import_guide(path: Path, front_axis: str, template_mesh):
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError("could not import I2-3D guide")
    imported = [item for item in bpy.data.objects if item not in before]
    meshes = [item for item in imported if item.type == "MESH"]
    if not meshes:
        raise RuntimeError("I2-3D guide contains no mesh")
    guide = max(meshes, key=lambda item: len(item.data.vertices))
    for item in imported:
        if item is not guide:
            bpy.data.objects.remove(item, do_unlink=True)
    if guide.data.uv_layers.active is None or not guide.material_slots:
        raise RuntimeError("I2-3D guide must preserve PBR material and UVs")

    yaw = CARDINAL_YAW_DEGREES[front_axis]
    guide.parent = None
    guide.matrix_world = Matrix.Rotation(math.radians(yaw), 4, "Z") @ guide.matrix_world
    bpy.context.view_layer.update()
    robust.align_bbox(guide, template_mesh, "uniform")
    bpy.ops.object.select_all(action="DESELECT")
    guide.select_set(True)
    bpy.context.view_layer.objects.active = guide
    robust.apply_transforms(guide)
    return guide, yaw


def majority_face_regions(vertex_regions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    counts = np.stack(
        [np.sum(vertex_regions[faces] == region, axis=1) for region in sorted(REGION_NAMES)],
        axis=1,
    )
    return np.argmax(counts, axis=1).astype(np.int64)


def region_bvhs(vertices: np.ndarray, faces: np.ndarray, face_regions: np.ndarray):
    vertices_list = [tuple(map(float, value)) for value in vertices]
    trees = {}
    for region in sorted(REGION_NAMES):
        indices = np.flatnonzero(face_regions == region)
        if len(indices) == 0:
            raise RuntimeError(f"guide has no faces for region {REGION_NAMES[region]}")
        polygons = [tuple(map(int, faces[index])) for index in indices]
        trees[region] = {
            "tree": BVHTree.FromPolygons(vertices_list, polygons, all_triangles=True),
            "face_indices": indices,
        }
    return trees


def vertex_adjacency(mesh):
    neighbors = [set() for _ in mesh.data.vertices]
    for edge in mesh.data.edges:
        first, second = map(int, edge.vertices)
        neighbors[first].add(second)
        neighbors[second].add(first)
    return tuple(tuple(sorted(values)) for values in neighbors)


def smooth_region_displacements(displacements, regions, adjacency, iterations, blend, locked):
    values = np.asarray(displacements, dtype=np.float64).copy()
    for _ in range(iterations):
        previous = values.copy()
        for index, neighbors in enumerate(adjacency):
            if locked[index]:
                continue
            compatible = [neighbor for neighbor in neighbors if regions[neighbor] == regions[index]]
            if compatible:
                mean = previous[compatible].mean(axis=0)
                values[index] = previous[index] * (1.0 - blend) + mean * blend
    values[locked] = 0.0
    return values


def closest_barycentric(point, triangle):
    # Reuse the tested closest-point implementation used by skin transfer.
    barycentric, _ = _closest_point_barycentric(
        np.asarray(point, dtype=np.float64),
        np.asarray(triangle[0], dtype=np.float64),
        np.asarray(triangle[1], dtype=np.float64),
        np.asarray(triangle[2], dtype=np.float64),
    )
    return barycentric


def guide_face_uvs(guide, faces):
    if any(len(polygon.vertices) != 3 for polygon in guide.data.polygons):
        raise RuntimeError("runtime guide must be triangulated")
    uv_layer = guide.data.uv_layers.active
    values = np.empty((len(faces), 3, 2), dtype=np.float64)
    for polygon in guide.data.polygons:
        values[polygon.index] = [
            tuple(uv_layer.data[loop_index].uv)
            for loop_index in polygon.loop_indices
        ]
    return values


def region_normalized_surface_matches(
    template_vertices,
    template_regions,
    guide_vertices,
    guide_regions,
    trees,
):
    """Match equivalent longitudinal/lateral positions inside each region."""
    matched_points = np.empty_like(template_vertices)
    matched_faces = np.full(len(template_vertices), -1, dtype=np.int64)
    for region in sorted(REGION_NAMES):
        template_indices = np.flatnonzero(template_regions == region)
        guide_indices = np.flatnonzero(guide_regions == region)
        if len(template_indices) == 0 or len(guide_indices) == 0:
            raise RuntimeError(
                f"empty normalized appearance region: {REGION_NAMES[region]}"
            )
        template_values = template_vertices[template_indices]
        guide_values = guide_vertices[guide_indices]
        template_minimum = template_values.min(axis=0)
        template_extent = np.ptp(template_values, axis=0)
        guide_minimum = guide_values.min(axis=0)
        guide_extent = np.ptp(guide_values, axis=0)
        normalized = (
            (template_values - template_minimum)
            / np.maximum(template_extent, 1.0e-9)
        )
        probes = guide_minimum + normalized * guide_extent
        entry = trees[int(region)]
        for template_index, probe in zip(template_indices, probes):
            nearest = entry["tree"].find_nearest(
                Vector(tuple(map(float, probe)))
            )
            if nearest is None or nearest[0] is None or nearest[2] is None:
                raise RuntimeError(
                    "no normalized guide match for "
                    f"{REGION_NAMES[int(region)]}"
                )
            location, _, local_face_index, _ = nearest
            matched_points[template_index] = tuple(location)
            matched_faces[template_index] = int(
                entry["face_indices"][int(local_face_index)]
            )
    if np.any(matched_faces < 0):
        raise RuntimeError("normalized appearance matching left vertices unmatched")
    return matched_points, matched_faces


def fit_surface_and_uv(
    template,
    guide,
    *,
    fit_strength,
    max_displacement_ratio,
    smooth_iterations,
    smooth_blend,
    foot_lock_height_ratio,
    geometry_fit_mode,
    project_guide_uv,
    transfer_vertex_color,
    region_normalized_texture,
):
    template_vertices = world_vertices(template)
    template_faces = triangulated_faces(template)
    template_region_vertices = robust.dog_region_coords(template_vertices, "positive-x")
    bounds = mesh_bounds(template_region_vertices)
    template_regions = coarse_region_labels(template_region_vertices, bounds=bounds)
    template_face_regions = face_region_labels(
        template_region_vertices, template_faces, bounds=bounds
    )

    guide_vertices = world_vertices(guide)
    guide_faces = triangulated_faces(guide)
    guide_region_vertices = robust.dog_region_coords(guide_vertices, "positive-x")
    coarse_guide_regions = coarse_region_labels(guide_region_vertices, bounds=bounds)
    guide_regions = target_region_labels_from_source_proximity(
        source_vertices=template_vertices,
        source_faces=template_faces,
        source_face_regions=template_face_regions,
        target_vertices=guide_vertices,
        coarse_target_regions=coarse_guide_regions,
    )
    guide_face_regions = majority_face_regions(guide_regions, guide_faces)
    trees = region_bvhs(guide_vertices, guide_faces, guide_face_regions)
    source_uv_triangles = (
        guide_face_uvs(guide, guide_faces)
        if project_guide_uv or transfer_vertex_color
        else None
    )

    matched_points = np.empty_like(template_vertices)
    matched_faces = np.full(len(template_vertices), -1, dtype=np.int64)
    distances = np.zeros(len(template_vertices), dtype=np.float64)
    for index, (point, region) in enumerate(zip(template_vertices, template_regions)):
        entry = trees[int(region)]
        nearest = entry["tree"].find_nearest(Vector(tuple(map(float, point))))
        if nearest is None or nearest[0] is None or nearest[2] is None:
            raise RuntimeError(f"no guide match for {REGION_NAMES[int(region)]}")
        location, _, local_face_index, distance = nearest
        matched_points[index] = tuple(location)
        matched_faces[index] = int(entry["face_indices"][int(local_face_index)])
        distances[index] = float(distance)

    raw_displacements = matched_points - template_vertices
    diagonal = float(np.linalg.norm(np.ptp(template_vertices, axis=0)))
    maximum = diagonal * max_displacement_ratio
    lengths = np.linalg.norm(raw_displacements, axis=1)
    scale = np.minimum(1.0, maximum / np.maximum(lengths, 1.0e-12))
    displacements = raw_displacements * scale[:, None] * fit_strength

    floor = float(template_vertices[:, 2].min())
    height = float(np.ptp(template_vertices[:, 2]))
    foot_locked = template_vertices[:, 2] <= floor + height * foot_lock_height_ratio
    limb_region_ids = {
        region
        for region, name in REGION_NAMES.items()
        if name in {
            "front_left_leg",
            "front_right_leg",
            "hind_left_leg",
            "hind_right_leg",
        }
    }
    limb_locked = (
        np.isin(template_regions, sorted(limb_region_ids))
        if geometry_fit_mode == "axial-only"
        else np.zeros(len(template_vertices), dtype=bool)
    )
    locked = np.logical_or(foot_locked, limb_locked)
    displacements = smooth_region_displacements(
        displacements,
        template_regions,
        vertex_adjacency(template),
        smooth_iterations,
        smooth_blend,
        locked,
    )
    fitted = template_vertices + displacements
    inverse = template.matrix_world.inverted()
    for vertex, value in zip(template.data.vertices, fitted):
        vertex.co = inverse @ Vector(tuple(map(float, value)))
    template.data.update()
    bpy.context.view_layer.update()

    transferred_vertex_uvs = None
    if project_guide_uv or transfer_vertex_color:
        appearance_matched_points = matched_points
        appearance_matched_faces = matched_faces
        if region_normalized_texture:
            (
                appearance_matched_points,
                appearance_matched_faces,
            ) = region_normalized_surface_matches(
                template_vertices,
                template_regions,
                guide_vertices,
                guide_regions,
                trees,
            )
        vertex_uvs = np.zeros((len(template_vertices), 2), dtype=np.float64)
        for index, face_index in enumerate(appearance_matched_faces):
            triangle = guide_vertices[guide_faces[face_index]]
            barycentric = closest_barycentric(
                appearance_matched_points[index], triangle
            )
            vertex_uvs[index] = barycentric @ source_uv_triangles[face_index]
        transferred_vertex_uvs = vertex_uvs
    if project_guide_uv:
        uv_layer = template.data.uv_layers.active
        if uv_layer is None:
            uv_layer = template.data.uv_layers.new(name="I23D_Projected_UV")
        for loop in template.data.loops:
            uv_layer.data[loop.index].uv = tuple(vertex_uvs[loop.vertex_index])

    return {
        "template_vertices": len(template_vertices),
        "template_faces": len(template_faces),
        "guide_vertices": len(guide_vertices),
        "guide_faces": len(guide_faces),
        "fit_strength": fit_strength,
        "geometry_fit_mode": geometry_fit_mode,
        "maximum_displacement_ratio": max_displacement_ratio,
        "maximum_allowed_displacement": maximum,
        "maximum_raw_match_distance": float(distances.max(initial=0.0)),
        "maximum_applied_displacement": float(
            np.linalg.norm(displacements, axis=1).max(initial=0.0)
        ),
        "mean_applied_displacement": float(
            np.linalg.norm(displacements, axis=1).mean()
        ),
        "locked_foot_vertices": int(foot_locked.sum()),
        "locked_limb_vertices": int(limb_locked.sum()),
        "limb_geometry_policy": (
            "preserve_subdivided_template_rest_geometry_and_native_weights"
            if geometry_fit_mode == "axial-only"
            else "fit_to_generated_guide"
        ),
        "appearance_correspondence_policy": (
            "semantic_region_normalized_longitudinal_lateral_nearest_surface"
            if region_normalized_texture
            else "semantic_region_world_space_nearest_surface"
        ),
        "uv_policy": (
            "nearest_guide_vertex_projection_legacy"
            if project_guide_uv
            else (
                "not_used_vertex_color_surface_sampling"
                if transfer_vertex_color
                else "preserve_template_uv_for_selected_to_active_bake"
            )
        ),
        "region_counts": {
            REGION_NAMES[region]: int(np.sum(template_regions == region))
            for region in sorted(REGION_NAMES)
        },
    }, transferred_vertex_uvs, template_regions


def realize_guide_textures(material, texture_dir):
    """Decode embedded WebP images to durable PNGs before glTF export.

    Blender imports EXT_texture_webp images as packed files with an empty
    filepath and often reports ``has_data=False``.  Copying that material and
    calling ``pack()`` can leave the glTF exporter's temporary image without
    pixel data.  A lossless PNG realization gives both Blender and the final
    exporter a stable, hashable source without changing the guide GLB.
    """
    texture_dir.mkdir(parents=True, exist_ok=False)
    records = []
    seen = set()
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image = node.image
        identity = int(image.as_pointer())
        if identity in seen:
            continue
        seen.add(identity)
        if len(image.pixels) <= 0:
            image.reload()
        if len(image.pixels) <= 0:
            raise RuntimeError(f"guide image did not decode: {image.name}")
        _ = image.pixels[0]
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in image.name
        ).strip("_") or "texture"
        path = texture_dir / f"{len(records):02d}_{safe_name}.png"
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace realized texture: {path}")
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"guide texture realization failed: {path}")
        records.append(
            {
                "image": image.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "size": [int(image.size[0]), int(image.size[1])],
                "source_format": "embedded_webp_or_imported_image",
                "realized_format": "PNG",
            }
        )
    if not records:
        raise RuntimeError("guide PBR material has no image textures")
    return records


def install_guide_material(template, guide, texture_dir):
    source_material = guide.material_slots[0].material
    if source_material is None or not source_material.use_nodes:
        raise RuntimeError("guide PBR material is missing")
    realized_texture_files = realize_guide_textures(source_material, texture_dir)
    material = source_material.copy()
    material.name = "I23D_Breed_Appearance"
    material.use_backface_culling = False
    template.data.materials.clear()
    template.data.materials.append(material)
    for polygon in template.data.polygons:
        polygon.material_index = 0
    images = []
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            images.append(
                {
                    "node": node.name,
                    "image": node.image.name,
                    "size": [int(node.image.size[0]), int(node.image.size[1])],
                }
            )
    if not images:
        raise RuntimeError("guide PBR material has no image textures")
    return {
        "backend": "projected-uv-legacy",
        "material": material.name,
        "images": images,
        "realized_texture_files": realized_texture_files,
    }


def guide_base_color_image(guide):
    material = guide.material_slots[0].material
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError("guide PBR material is missing")
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is None:
        raise RuntimeError("guide PBR material has no Principled BSDF")
    base_color = principled.inputs.get("Base Color")
    if base_color is None or not base_color.is_linked:
        raise RuntimeError("guide PBR Base Color is not texture-backed")
    source_node = base_color.links[0].from_node
    if source_node.type != "TEX_IMAGE" or source_node.image is None:
        raise RuntimeError("guide PBR Base Color must link directly from an image")
    image = source_node.image
    if len(image.pixels) <= 0:
        image.reload()
    if len(image.pixels) <= 0:
        raise RuntimeError(f"guide base-color image did not decode: {image.name}")
    return image


def bilinear_sample_image(image, uvs):
    width, height = map(int, image.size)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape((height, width, 4))
    values = np.asarray(uvs, dtype=np.float64)
    u = np.mod(values[:, 0], 1.0) * (width - 1)
    v = np.mod(values[:, 1], 1.0) * (height - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    first = pixels[y0, x0, :3] * (1.0 - fx) + pixels[y0, x1, :3] * fx
    second = pixels[y1, x0, :3] * (1.0 - fx) + pixels[y1, x1, :3] * fx
    return first * (1.0 - fy) + second * fy


def smooth_vertex_colors(colors, regions, adjacency, iterations=3, blend=0.45):
    values = np.asarray(colors, dtype=np.float32).copy()
    for _ in range(iterations):
        previous = values.copy()
        for index, neighbors in enumerate(adjacency):
            compatible = [
                neighbor for neighbor in neighbors if regions[neighbor] == regions[index]
            ]
            if compatible:
                values[index] = (
                    previous[index] * (1.0 - blend)
                    + previous[compatible].mean(axis=0) * blend
                )
    return np.clip(values, 0.0, 1.0)


def install_guide_vertex_color_material(template, guide, vertex_uvs, regions):
    """Sample generated PBR color without introducing a new mesh or UV seams."""
    if vertex_uvs is None or len(vertex_uvs) != len(template.data.vertices):
        raise RuntimeError("vertex-color transfer requires one guide UV per template vertex")
    image = guide_base_color_image(guide)
    colors = bilinear_sample_image(image, vertex_uvs)
    colors = smooth_vertex_colors(
        colors,
        regions,
        vertex_adjacency(template),
    )
    rgba = np.concatenate(
        [colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1
    )
    attribute = template.data.color_attributes.new(
        name="I23D_Breed_VertexColor",
        type="BYTE_COLOR",
        domain="CORNER",
    )
    loop_vertices = np.asarray(
        [loop.vertex_index for loop in template.data.loops], dtype=np.int64
    )
    attribute.data.foreach_set("color", rgba[loop_vertices].reshape(-1))

    template.data.materials.clear()
    material = bpy.data.materials.new(name="I23D_Breed_Appearance_VertexColor")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = attribute.name
    material.node_tree.links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])
    material.node_tree.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.82
    template.data.materials.append(material)
    for polygon in template.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True
    return {
        "backend": "nearest_surface_region_vertex_color_v1",
        "material": material.name,
        "attribute": attribute.name,
        "attribute_domain": "CORNER",
        "attribute_storage": "BYTE_COLOR",
        "source_image": image.name,
        "source_image_size": [int(image.size[0]), int(image.size[1])],
        "template_vertices": len(template.data.vertices),
        "template_loops": len(template.data.loops),
        "smoothing": {"iterations": 3, "blend": 0.45, "within_region_only": True},
        "metallic_policy": "constant_zero_for_nonmetallic_animal_surface",
        "roughness_policy": "constant_0.82",
    }


def bake_region_sampled_guide_to_template(
    template,
    guide,
    vertex_uvs,
    regions,
    *,
    resolution,
):
    """Bake semantic nearest-surface colours onto the stable template UV.

    A raw selected-to-active bake follows target normals and can hit the
    opposite side, another limb, or empty space when generated and template
    silhouettes differ. The supplied vertex UVs already came from a nearest
    guide triangle in the same semantic body/limb region, so use those samples
    as the correspondence authority and rasterize them on the template itself.
    """
    if vertex_uvs is None or len(vertex_uvs) != len(template.data.vertices):
        raise RuntimeError(
            "region-atlas transfer requires one region-constrained guide UV "
            "per template vertex"
        )
    source_image = guide_base_color_image(guide)
    colors = bilinear_sample_image(source_image, vertex_uvs)
    colors = smooth_vertex_colors(
        colors,
        regions,
        vertex_adjacency(template),
        iterations=0,
        blend=0.0,
    )
    rgba = np.concatenate(
        [colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1
    )
    attribute = template.data.color_attributes.new(
        name="I23D_Region_Sampled_BaseColor",
        type="FLOAT_COLOR",
        domain="CORNER",
    )
    loop_vertices = np.asarray(
        [loop.vertex_index for loop in template.data.loops], dtype=np.int64
    )
    attribute.data.foreach_set("color", rgba[loop_vertices].reshape(-1))
    uv_atlas = ensure_template_bake_uv(template)

    template.data.materials.clear()
    material = bpy.data.materials.new(name="I23D_Region_Atlas_Bake_Source")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = attribute.name
    target_image = new_bake_image(
        "I23D_Region_Atlas_BaseColor", resolution, "sRGB"
    )
    target_texture = nodes.new("ShaderNodeTexImage")
    target_texture.name = "Region Atlas Bake Target"
    target_texture.image = target_image
    nodes.active = target_texture
    material.node_tree.links.new(
        vertex_color.outputs["Color"], emission.inputs["Color"]
    )
    material.node_tree.links.new(
        emission.outputs["Emission"], output_node.inputs["Surface"]
    )
    template.data.materials.append(material)
    for polygon in template.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    bpy.ops.object.select_all(action="DESELECT")
    template.select_set(True)
    bpy.context.view_layer.objects.active = template
    print(
        "STABLE_TEMPLATE_STAGE region_atlas_base_color_start "
        f"resolution={resolution}",
        flush=True,
    )
    bpy.ops.object.bake(
        type="EMIT",
        use_selected_to_active=False,
        use_clear=True,
        margin=16,
    )
    target_image.pack()
    print("STABLE_TEMPLATE_STAGE region_atlas_base_color_done", flush=True)

    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "Region Sampled Base Color"
    texture.image = target_image
    material.node_tree.links.new(
        texture.outputs["Color"], principled.inputs["Base Color"]
    )
    material.node_tree.links.new(
        principled.outputs["BSDF"], output_node.inputs["Surface"]
    )
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.82
    material.name = "I23D_Breed_Appearance_Region_Atlas"
    return {
        "backend": "semantic_nearest_surface_template_uv_atlas_v1",
        "material": material.name,
        "resolution": resolution,
        "image": target_image.name,
        "source_image": source_image.name,
        "source_image_size": [
            int(source_image.size[0]),
            int(source_image.size[1]),
        ],
        "correspondence_policy": (
            "nearest_guide_triangle_within_matching_semantic_region"
        ),
        "rasterization_policy": "self_emission_bake_to_template_uv",
        "smoothing": {
            "iterations": 0,
            "blend": 0.0,
            "within_region_only": True,
        },
        "uv_atlas": uv_atlas,
        "metallic_policy": "constant_zero_for_nonmetallic_animal_surface",
        "roughness_policy": "constant_0.82",
    }


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
    """Undo Cycles storing encoded sRGB samples as scene-linear values."""
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))
    encoded = np.clip(pixels[:, :3], 0.0, 1.0)
    pixels[:, :3] = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()


def select_bake_pair(source, target):
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target


def ensure_template_bake_uv(template):
    """Create a deterministic atlas when the low-poly carrier has no UVs."""
    if template.data.uv_layers.active is not None:
        return {
            "source": "template_existing",
            "name": template.data.uv_layers.active.name,
        }
    bpy.ops.object.select_all(action="DESELECT")
    template.select_set(True)
    bpy.context.view_layer.objects.active = template
    template.data.uv_layers.new(name="Stable_Template_Bake_UV")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=1.1519173063162575,
        island_margin=0.02,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    return {
        "source": "deterministic_smart_project",
        "name": template.data.uv_layers.active.name,
    }


def bake_guide_material_to_template(
    template,
    guide,
    *,
    resolution,
    max_ray_distance_ratio,
):
    """Bake guide PBR to the stable template's own continuous UV atlas."""
    if not 0.01 <= max_ray_distance_ratio <= 0.30:
        raise RuntimeError("bake max ray distance ratio must be in [0.01, 0.30]")
    uv_atlas = ensure_template_bake_uv(template)

    template.data.materials.clear()
    material = bpy.data.materials.new(name="I23D_Breed_Appearance_Baked")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    material.node_tree.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
    template.data.materials.append(material)
    for polygon in template.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    diagonal = float(np.linalg.norm(np.ptp(world_vertices(template), axis=0)))
    max_ray_distance = max(diagonal * max_ray_distance_ratio, 1.0e-6)
    cage_extrusion = max(diagonal * 0.005, 1.0e-6)

    base_image = new_bake_image("I23D_Baked_BaseColor", resolution, "sRGB")
    base_texture = nodes.new("ShaderNodeTexImage")
    base_texture.name = "Baked Base Color"
    base_texture.image = base_image
    nodes.active = base_texture
    select_bake_pair(guide, template)
    print(
        "STABLE_TEMPLATE_STAGE pbr_bake_base_color_start "
        f"resolution={resolution} max_ray_distance={max_ray_distance:.8f}",
        flush=True,
    )
    bpy.ops.object.bake(
        type="DIFFUSE",
        pass_filter={"COLOR"},
        use_selected_to_active=True,
        use_clear=True,
        margin=16,
        cage_extrusion=cage_extrusion,
        max_ray_distance=max_ray_distance,
    )
    reinterpret_baked_srgb_values_as_scene_linear(base_image)
    base_image.pack()
    print("STABLE_TEMPLATE_STAGE pbr_bake_base_color_done", flush=True)

    roughness_image = new_bake_image(
        "I23D_Baked_Roughness", resolution, "Non-Color"
    )
    roughness_texture = nodes.new("ShaderNodeTexImage")
    roughness_texture.name = "Baked Roughness"
    roughness_texture.image = roughness_image
    nodes.active = roughness_texture
    select_bake_pair(guide, template)
    print("STABLE_TEMPLATE_STAGE pbr_bake_roughness_start", flush=True)
    bpy.ops.object.bake(
        type="ROUGHNESS",
        use_selected_to_active=True,
        use_clear=True,
        margin=16,
        cage_extrusion=cage_extrusion,
        max_ray_distance=max_ray_distance,
    )
    roughness_image.pack()
    material.node_tree.links.new(
        base_texture.outputs["Color"], principled.inputs["Base Color"]
    )
    material.node_tree.links.new(
        roughness_texture.outputs["Color"], principled.inputs["Roughness"]
    )
    principled.inputs["Metallic"].default_value = 0.0
    print("STABLE_TEMPLATE_STAGE pbr_bake_roughness_done", flush=True)
    return {
        "backend": "cycles_selected_to_active_template_uv_v1",
        "material": material.name,
        "resolution": resolution,
        "device": "CPU",
        "max_ray_distance_ratio": max_ray_distance_ratio,
        "max_ray_distance": max_ray_distance,
        "cage_extrusion": cage_extrusion,
        "images": [base_image.name, roughness_image.name],
        "uv_atlas": uv_atlas,
        "metallic_policy": "constant_zero_for_nonmetallic_animal_surface",
        "base_color_encoding_policy": "srgb_to_scene_linear_after_bake",
    }


def remove_guide(guide):
    data = guide.data
    bpy.data.objects.remove(guide, do_unlink=True)
    if data.users == 0:
        bpy.data.meshes.remove(data)


def export_template(template, armature, output):
    robust.keep_canonical_walk_idle_actions(armature)
    bpy.ops.object.select_all(action="DESELECT")
    template.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_extra_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    robust.postprocess_glb_animation_channels(
        output,
        {"translation", "rotation"},
        canonical_walk_idle=True,
    )


def main():
    args = parse_argv()
    if not 0.0 < args.fit_strength <= 1.0:
        raise SystemExit("--fit-strength must be in (0, 1]")
    if not 0.0 < args.max_displacement_ratio <= 0.25:
        raise SystemExit("--max-displacement-ratio must be in (0, 0.25]")
    if not 0 <= args.smooth_iterations <= 20 or not 0.0 <= args.smooth_blend <= 1.0:
        raise SystemExit("invalid smoothing parameters")
    template_path = require_input(args.template_glb, "template")
    guide_path = require_input(args.guide_glb, "guide")
    output_path = require_new_output(args.output_glb, "output GLB")
    manifest_path = require_new_output(args.manifest, "manifest")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(template_path))
    template, armature, input_actions = identify_template()
    subdivision = apply_template_subdivision(template, args.subdivision_levels)
    guide, cardinal_yaw = import_guide(
        guide_path, args.guide_front_axis, template
    )
    fit, transferred_vertex_uvs, template_regions = fit_surface_and_uv(
        template,
        guide,
        fit_strength=args.fit_strength,
        max_displacement_ratio=args.max_displacement_ratio,
        smooth_iterations=args.smooth_iterations,
        smooth_blend=args.smooth_blend,
        foot_lock_height_ratio=args.foot_lock_height_ratio,
        geometry_fit_mode=args.geometry_fit_mode,
        project_guide_uv=args.appearance_transfer == "projected-uv",
        transfer_vertex_color=args.appearance_transfer
        in {"vertex-color", "region-atlas"},
        region_normalized_texture=args.appearance_transfer == "region-atlas",
    )
    if args.appearance_transfer == "vertex-color":
        material = install_guide_vertex_color_material(
            template,
            guide,
            transferred_vertex_uvs,
            template_regions,
        )
    elif args.appearance_transfer == "region-atlas":
        material = bake_region_sampled_guide_to_template(
            template,
            guide,
            transferred_vertex_uvs,
            template_regions,
            resolution=args.bake_resolution,
        )
    elif args.appearance_transfer == "bake":
        material = bake_guide_material_to_template(
            template,
            guide,
            resolution=args.bake_resolution,
            max_ray_distance_ratio=args.bake_max_ray_distance_ratio,
        )
    else:
        material = install_guide_material(
            template,
            guide,
            output_path.parent / "realized_guide_textures",
        )
    remove_guide(guide)
    export_template(template, armature, output_path)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "template": {
            "path": str(template_path),
            "sha256": sha256_file(template_path),
            "size_bytes": template_path.stat().st_size,
            "front_axis": "positive-x",
            "actions": input_actions,
        },
        "guide": {
            "path": str(guide_path),
            "sha256": sha256_file(guide_path),
            "size_bytes": guide_path.stat().st_size,
            "front_axis": args.guide_front_axis,
            "cardinal_yaw_to_positive_x_degrees": cardinal_yaw,
            "fine_yaw_inference": False,
        },
        "subdivision": subdivision,
        "surface_fit": fit,
        "material_projection": material,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "instance_policy": {
            "rerig_per_colour_instance": False,
            "breed_template_fit_frequency": "once_per_approved_breed_template",
            "colour_instances": "bake_approved_FLUX_PBR_onto_frozen_template_UV",
            "size_instances": "apply_one_of_three_validated_actor_scales",
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "STABLE_ANIMAL_BREED_TEMPLATE_FIT_OK "
        f"output={output_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
