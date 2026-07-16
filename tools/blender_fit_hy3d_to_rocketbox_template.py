#!/usr/bin/env python3

"""Fit an authenticated 3D appearance guide onto stable Rocketbox topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import blender_bind_hy3d_to_rocketbox as direct
import blender_render_rocketbox_source_review as source_review
import human_template_fit
import i23d_rocketbox_contract
from tools.human_part_transfer import (
    HumanRegion,
    source_face_regions,
    source_vertex_regions_from_weights,
    target_regions_from_capsules,
)


BINDING_MODE = "stable_rocketbox_template_fit_v1"
USAGE_SCOPE = "technical_spike_only"
I23D_BAKEOFF_ROOT = SPEAR_ROOT / "tmp" / "i23d_human_bakeoff_v1"
FIT_STRENGTH = 0.35
FIT_MAX_HEIGHT_RATIO = 0.035
FIT_SMOOTH_ITERATIONS = 4
FIT_SMOOTH_BLEND = 0.5
TEXTURE_SIZE = 1024
TEXTURE_DILATE_ITERATIONS = 6
MAX_ACTION_FLOOR_OFFSET_M = 0.05
MAX_POST_NORMALIZE_PENETRATION_M = 0.01
FOOT_SUPPORT_TOLERANCE_M = 0.015
OUTPUT_FILENAMES = (
    "cleaned.obj",
    "bound.blend",
    "bound_walk.glb",
    "bound_idle.glb",
    "template_fit_metrics.json",
    "bind_metrics.json",
    "bind_manifest.json",
    "reference.png",
)
TEXTURE_PREFIXES = {
    "rocketbox_male_adult_01": "m002",
    "rocketbox_female_adult_01": "f001",
}
TEXTURE_DIRECTORIES = {
    "rocketbox_male_adult_01": Path(
        "/data/datasets/rocketbox/sample/Assets/Avatars/Adults/"
        "Male_Adult_01/Textures"
    ),
    "rocketbox_female_adult_01": Path(
        "/data/datasets/rocketbox/sample/Assets/Avatars/Adults/"
        "Female_Adult_01/Textures"
    ),
}
PBR_ROLES = ("diffuse", "metallic", "roughness")
SURFACE_MATERIAL_ROLES = ("body", "head")
FOOT_GROUPS = {
    "left": ("Bip01 L Foot", "Bip01 L Toe0"),
    "right": ("Bip01 R Foot", "Bip01 R Toe0"),
}


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    guide_group = parser.add_mutually_exclusive_group(required=True)
    guide_group.add_argument("--hy3d-dir", type=Path)
    guide_group.add_argument("--guide-glb", type=Path)
    parser.add_argument("--guide-manifest", type=Path)
    parser.add_argument(
        "--guide-backend", choices=tuple(i23d_rocketbox_contract.BACKEND_CONTRACTS)
    )
    parser.add_argument("--front-axis", choices=("negative-y", "positive-y"))
    parser.add_argument("--reference-rgba", type=Path)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.asset_id not in direct.EXPECTED_ASSET_IDS:
        parser.error(f"unexpected Rocketbox asset id: {args.asset_id}")
    i23d_fields = (
        args.guide_manifest,
        args.guide_backend,
        args.front_axis,
        args.reference_rgba,
    )
    if args.guide_glb is not None and any(value is None for value in i23d_fields):
        parser.error(
            "--guide-glb requires --guide-manifest, --guide-backend, "
            "--front-axis, and --reference-rgba"
        )
    if args.hy3d_dir is not None and any(value is not None for value in i23d_fields):
        parser.error("I23D guide options cannot be combined with --hy3d-dir")
    return args


def validate_i23d_inputs(args):
    expected_dir = I23D_BAKEOFF_ROOT / args.guide_backend / args.asset_id
    expected_glb = expected_dir / "canary_1024_seed42.glb"
    expected_manifest = expected_dir / "canary_1024_seed42.manifest.json"
    expected_reference = (
        I23D_BAKEOFF_ROOT
        / "inputs"
        / args.asset_id
        / "input_rgba_isnet.png"
    )
    guide_glb = direct.require_regular_file(args.guide_glb, "I23D guide GLB")
    guide_manifest_path = direct.require_regular_file(
        args.guide_manifest, "I23D guide manifest"
    )
    reference_rgba = direct.require_regular_file(
        args.reference_rgba, "I23D reference RGBA"
    )
    expected = {
        "guide GLB": expected_glb.absolute(),
        "guide manifest": expected_manifest.absolute(),
        "reference RGBA": expected_reference.absolute(),
    }
    actual = {
        "guide GLB": guide_glb,
        "guide manifest": guide_manifest_path,
        "reference RGBA": reference_rgba,
    }
    for description, expected_path in expected.items():
        if actual[description] != expected_path:
            raise ValueError(f"{description} must be the reviewed canary: {expected_path}")
    manifest = direct.load_json_object(guide_manifest_path, "I23D guide manifest")
    provenance = i23d_rocketbox_contract.validate_i23d_manifest(
        manifest,
        asset_id=args.asset_id,
        backend=args.guide_backend,
        front_axis=args.front_axis,
        glb_path=guide_glb,
        reference_path=reference_rgba,
    )
    return {
        "guide_glb_path": guide_glb,
        "guide_manifest_path": guide_manifest_path,
        "guide_manifest": manifest,
        "reference_rgba_path": reference_rgba,
        "provenance": provenance,
    }


def authenticated_i23d_snapshot_sources(baseline, guide, idle):
    return {
        "baseline_manifest": {
            "path": baseline["baseline_manifest_path"],
            "filename": "baseline_manifest.json",
            "sha256": baseline["baseline_manifest_sha256"],
            "size_bytes": baseline["baseline_manifest_size"],
        },
        "baseline_blend": {
            "path": baseline["baseline_blend_path"],
            "filename": "retarget.blend",
            "sha256": baseline["baseline_blend_sha256"],
            "size_bytes": baseline["baseline_blend_size"],
        },
        "i23d_manifest": {
            "path": guide["guide_manifest_path"],
            "filename": "i23d_manifest.json",
            "sha256": direct.sha256_file(guide["guide_manifest_path"]),
            "size_bytes": guide["guide_manifest_path"].stat().st_size,
        },
        "i23d_guide_glb": {
            "path": guide["guide_glb_path"],
            "filename": "i23d_guide.glb",
            "sha256": guide["provenance"]["guide_glb"]["sha256"],
            "size_bytes": guide["provenance"]["guide_glb"]["size_bytes"],
        },
        "i23d_reference": {
            "path": guide["reference_rgba_path"],
            "filename": "reference.png",
            "sha256": guide["provenance"]["reference_rgba"]["sha256"],
            "size_bytes": guide["provenance"]["reference_rgba"]["size_bytes"],
        },
        "idle_motion_fbx": {
            "path": idle["idle_motion_fbx_path"],
            "filename": idle["idle_motion_fbx_path"].name,
            "sha256": idle["idle_motion_fbx_sha256"],
            "size_bytes": idle["idle_motion_fbx_size"],
        },
    }


def stage_i23d_input_snapshot(output_dir, baseline, guide, idle):
    output_dir = Path(output_dir).absolute()
    snapshot_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.bind-inputs.",
            dir=output_dir.parent,
        )
    )
    os.chmod(snapshot_root, 0o700)
    paths = {}
    records = {}
    try:
        for label, source in authenticated_i23d_snapshot_sources(
            baseline, guide, idle
        ).items():
            destination = snapshot_root / source["filename"]
            records[label] = direct.copy_authenticated_file(
                source["path"],
                destination,
                source["sha256"],
                source["size_bytes"],
            )
            paths[label] = destination
        records["idle_motion_fbx"]["git_blob_sha1"] = idle[
            "idle_motion_fbx_git_blob_sha1"
        ]
        return {"root": snapshot_root, "paths": paths, "records": records}
    except BaseException:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def capture_i23d_source_hashes(baseline, guide, idle):
    return {
        "baseline_manifest_sha256": baseline["baseline_manifest_sha256"],
        "baseline_blend_sha256": baseline["baseline_blend_sha256"],
        "i23d_manifest_sha256": direct.sha256_file(guide["guide_manifest_path"]),
        "i23d_guide_glb_sha256": guide["provenance"]["guide_glb"]["sha256"],
        "i23d_reference_rgba_sha256": guide["provenance"]["reference_rgba"][
            "sha256"
        ],
        "idle_motion_fbx_sha256": idle["idle_motion_fbx_sha256"],
    }


def verify_i23d_source_hashes_current(baseline, guide, idle, captured):
    current = {
        "baseline_manifest_current_sha256": direct.sha256_file(
            baseline["baseline_manifest_path"]
        ),
        "baseline_blend_current_sha256": direct.sha256_file(
            baseline["baseline_blend_path"]
        ),
        "i23d_manifest_current_sha256": direct.sha256_file(
            guide["guide_manifest_path"]
        ),
        "i23d_guide_glb_current_sha256": direct.sha256_file(
            guide["guide_glb_path"]
        ),
        "i23d_reference_rgba_current_sha256": direct.sha256_file(
            guide["reference_rgba_path"]
        ),
        "idle_motion_fbx_current_sha256": direct.sha256_file(
            idle["idle_motion_fbx_path"]
        ),
    }
    comparisons = (
        (captured["baseline_manifest_sha256"], current["baseline_manifest_current_sha256"]),
        (captured["baseline_blend_sha256"], current["baseline_blend_current_sha256"]),
        (captured["i23d_manifest_sha256"], current["i23d_manifest_current_sha256"]),
        (captured["i23d_guide_glb_sha256"], current["i23d_guide_glb_current_sha256"]),
        (
            captured["i23d_reference_rgba_sha256"],
            current["i23d_reference_rgba_current_sha256"],
        ),
        (captured["idle_motion_fbx_sha256"], current["idle_motion_fbx_current_sha256"]),
    )
    if any(first != second for first, second in comparisons):
        raise RuntimeError("I23D source inputs changed during template fitting")
    return current


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def int_array_sha256(values):
    array = np.ascontiguousarray(values, dtype=np.int64)
    return sha256_bytes(array.tobytes())


def polygon_index_payload(mesh):
    payload = []
    for polygon in mesh.data.polygons:
        payload.append(len(polygon.vertices))
        payload.extend(int(index) for index in polygon.vertices)
    return np.asarray(payload, dtype=np.int64)


def capture_template_contract(mesh, armature):
    weight_payload = []
    for vertex in mesh.data.vertices:
        weight_payload.append(
            tuple(
                sorted(
                    (int(item.group), round(float(item.weight), 12))
                    for item in vertex.groups
                    if item.weight > 0.0
                )
            )
        )
    return {
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "loop_vertex_indices_sha256": int_array_sha256(
            [loop.vertex_index for loop in mesh.data.loops]
        ),
        "polygon_vertex_indices_sha256": int_array_sha256(
            polygon_index_payload(mesh)
        ),
        "skin_contract": {
            "group_names": [group.name for group in mesh.vertex_groups],
            "weights_sha256": sha256_bytes(
                json.dumps(weight_payload, separators=(",", ":")).encode("ascii")
            ),
        },
        "uv_layers": [layer.name for layer in mesh.data.uv_layers],
        "material_slots": [
            slot.material.name if slot.material is not None else None
            for slot in mesh.material_slots
        ],
        "armature_bones": [bone.name for bone in armature.data.bones],
    }


def validate_template_contract(before, mesh, armature):
    after = capture_template_contract(mesh, armature)
    if before != after:
        raise RuntimeError("stable Rocketbox template contract changed")
    return after


def mesh_vertices_in_armature_space(mesh, armature):
    transform = armature.matrix_world.inverted() @ mesh.matrix_world
    return np.asarray(
        [tuple(transform @ vertex.co) for vertex in mesh.data.vertices],
        dtype=np.float64,
    )


def set_mesh_vertices_from_armature_space(mesh, armature, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(mesh.data.vertices), 3):
        raise RuntimeError("fitted template vertex array shape changed")
    transform = mesh.matrix_world.inverted() @ armature.matrix_world
    for vertex, value in zip(mesh.data.vertices, values):
        vertex.co = transform @ Vector(tuple(map(float, value)))
    mesh.data.update()
    bpy.context.view_layer.update()


def opacity_vertex_mask(mesh):
    opacity_slots = {
        index
        for index, slot in enumerate(mesh.material_slots)
        if slot.material is not None and "opacity" in slot.material.name.lower()
    }
    mask = np.zeros(len(mesh.data.vertices), dtype=bool)
    for polygon in mesh.data.polygons:
        if polygon.material_index in opacity_slots:
            mask[np.asarray(polygon.vertices, dtype=np.int64)] = True
    return mask


def vertex_adjacency(mesh):
    neighbors = [set() for _ in mesh.data.vertices]
    for edge in mesh.data.edges:
        first, second = map(int, edge.vertices)
        neighbors[first].add(second)
        neighbors[second].add(first)
    return tuple(tuple(sorted(values)) for values in neighbors)


def region_face_trees(vertices, faces, vertex_regions):
    face_regions = source_face_regions(faces, vertex_regions)
    trees = {}
    for region in HumanRegion:
        face_indices = np.flatnonzero(face_regions == int(region))
        if len(face_indices) == 0:
            raise RuntimeError(f"Hunyuan guide has no faces for region {region.name}")
        polygons = [tuple(map(int, faces[index])) for index in face_indices]
        tree = BVHTree.FromPolygons(
            [tuple(map(float, value)) for value in vertices],
            polygons,
            all_triangles=True,
        )
        trees[region] = {
            "tree": tree,
            "face_indices": face_indices,
        }
    return trees, face_regions


def find_region_nearest(point, region, region_trees):
    entry = region_trees[HumanRegion(int(region))]
    nearest = entry["tree"].find_nearest(Vector(tuple(map(float, point))))
    if nearest is None or nearest[0] is None or nearest[2] is None:
        raise RuntimeError(f"no Hunyuan surface match for region {HumanRegion(int(region)).name}")
    location, normal, local_face_index, distance = nearest
    global_face_index = int(entry["face_indices"][int(local_face_index)])
    return (
        np.asarray(tuple(location), dtype=np.float64),
        np.asarray(tuple(normal), dtype=np.float64),
        global_face_index,
        float(distance),
    )


def fit_template_surface(
    source_mesh,
    armature,
    source,
    guide_vertices,
    guide_faces,
    guide_regions,
):
    template_vertices = mesh_vertices_in_armature_space(source_mesh, armature)
    template_regions = source_vertex_regions_from_weights(
        source["weights"], source["group_names"]
    )
    region_trees, face_regions = region_face_trees(
        guide_vertices, guide_faces, guide_regions
    )
    opacity_mask = opacity_vertex_mask(source_mesh)
    displacements = np.zeros_like(template_vertices)
    distances = np.zeros(len(template_vertices), dtype=np.float64)
    matched_faces = np.full(len(template_vertices), -1, dtype=np.int64)
    for index, (point, region) in enumerate(zip(template_vertices, template_regions)):
        if opacity_mask[index]:
            continue
        nearest, _, face_index, distance = find_region_nearest(
            point, region, region_trees
        )
        displacements[index] = nearest - point
        distances[index] = distance
        matched_faces[index] = face_index
    displacements[:, 2] = 0.0
    height = float(np.ptp(template_vertices[:, 2]))
    max_distance = height*FIT_MAX_HEIGHT_RATIO
    displacements = human_template_fit.clamp_xy_displacements(
        displacements, max_distance
    )
    displacements *= FIT_STRENGTH
    displacements = human_template_fit.smooth_xy_displacements(
        displacements,
        vertex_adjacency(source_mesh),
        opacity_mask,
        iterations=FIT_SMOOTH_ITERATIONS,
        blend=FIT_SMOOTH_BLEND,
    )
    fitted = template_vertices + displacements
    fitted[:, 2] = template_vertices[:, 2]
    set_mesh_vertices_from_armature_space(source_mesh, armature, fitted)
    return {
        "template_vertices": fitted,
        "template_regions": template_regions,
        "guide_region_trees": region_trees,
        "guide_face_regions": face_regions,
        "matched_faces": matched_faces,
        "opacity_vertex_mask": opacity_mask,
        "metrics": {
            "fit_strength": FIT_STRENGTH,
            "max_height_ratio": FIT_MAX_HEIGHT_RATIO,
            "max_distance": max_distance,
            "fixed_opacity_vertices": int(opacity_mask.sum()),
            "maximum_raw_distance": float(distances.max(initial=0.0)),
            "maximum_applied_xy_displacement": float(
                np.linalg.norm(displacements[:, :2], axis=1).max(initial=0.0)
            ),
            "maximum_z_change": float(
                np.abs(fitted[:, 2] - template_vertices[:, 2]).max(initial=0.0)
            ),
            "region_counts": {
                HumanRegion(int(region)).name.lower(): int(
                    np.sum(template_regions == int(region))
                )
                for region in HumanRegion
            },
        },
    }


def import_clean_hy3d_guide(path, asset_id, source, pbr_paths):
    contract = direct.RAW_HY3D_AXIS_CONTRACTS[asset_id]
    before = set(bpy.data.objects)
    result = bpy.ops.wm.obj_import(
        filepath=str(path),
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )
    if "FINISHED" not in result:
        raise RuntimeError("could not import Hunyuan guide OBJ")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(imported) != 1 or len(meshes) != 1:
        raise RuntimeError("Hunyuan guide must import as one mesh")
    guide = meshes[0]
    guide.name = "Hunyuan_Appearance_Guide"
    direct.triangulate_mesh(guide)
    axis_contract = {
        **contract,
        **direct.validate_import_basis_matrix(
            np.asarray(guide.matrix_world.to_3x3(), dtype=np.float64), contract
        ),
        "raw_extents": np.ptp(
            np.asarray([tuple(vertex.co) for vertex in guide.data.vertices]), axis=0
        ).tolist(),
        "target_rotate_z_deg": direct.TARGET_ROTATE_Z_DEG,
    }
    import_cleanup = direct.cleanup_import_ground_artifacts(guide)
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    direct.move_target_to_armature_space(guide, armature)
    guide_regions, cleanup = direct.cleanup_target_geometry(guide, source)
    pbr = direct.assign_hunyuan_pbr_material(guide, pbr_paths)
    vertices, faces = direct.mesh_arrays(guide)
    return guide, vertices, faces, guide_regions, {
        "axis_contract": axis_contract,
        "import_cleanup": import_cleanup,
        "cleanup": cleanup,
        "pbr": pbr,
    }


def import_clean_i23d_guide(path, args, source):
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError("could not import authenticated I23D guide GLB")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError("I23D guide GLB must import as exactly one mesh")
    guide = meshes[0]
    guide.name = "I23D_Appearance_Guide"

    yaw_radians = math.pi if args.front_axis == "positive-y" else 0.0
    canonical_world = Matrix.Rotation(yaw_radians, 4, "Z") @ guide.matrix_world
    guide.parent = None
    guide.matrix_world = canonical_world
    for imported_object in imported:
        if imported_object != guide:
            bpy.data.objects.remove(imported_object, do_unlink=True)
    bpy.context.view_layer.update()

    if any(len(polygon.vertices) != 3 for polygon in guide.data.polygons):
        direct.triangulate_mesh(guide)
    if len(guide.data.uv_layers) < 1 or len(guide.material_slots) != 1:
        raise RuntimeError("I23D guide must preserve one textured material and UVs")
    material = guide.material_slots[0].material
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError("I23D guide material must contain a glTF PBR node tree")

    canonical_vertices = np.asarray(
        [tuple(guide.matrix_world @ vertex.co) for vertex in guide.data.vertices],
        dtype=np.float64,
    )
    source_front = np.array(
        (0.0, -1.0 if args.front_axis == "negative-y" else 1.0, 0.0),
        dtype=np.float64,
    )
    yaw_basis = np.asarray(Matrix.Rotation(yaw_radians, 3, "Z"), dtype=np.float64)
    canonical_front = yaw_basis @ source_front
    if not np.allclose(canonical_front, (0.0, -1.0, 0.0), atol=1.0e-7):
        raise RuntimeError("I23D front-axis normalization did not produce canonical -Y")
    axis_contract = {
        "source_front_axis": args.front_axis,
        "source_up_axis": "positive-z-after-gltf-import",
        "canonical_front_axis": "negative-y",
        "canonical_up_axis": "positive-z",
        "canonical_yaw_deg": 180.0 if yaw_radians else 0.0,
        "canonical_front_vector": canonical_front.tolist(),
        "basis_determinant": float(np.linalg.det(yaw_basis)),
        "raw_extents_after_gltf_import": np.ptp(
            canonical_vertices, axis=0
        ).tolist(),
    }

    import_cleanup = direct.cleanup_import_ground_artifacts(guide)
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    direct.move_target_to_armature_space(guide, armature)
    guide_regions, cleanup = direct.cleanup_target_geometry(guide, source)
    vertices, faces = direct.mesh_arrays(guide)
    image_nodes = [
        node
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    ]
    pbr = {
        "material_name": material.name,
        "material_slot_count": len(guide.material_slots),
        "uv_layer_count": len(guide.data.uv_layers),
        "image_names": [node.image.name for node in image_nodes],
        "image_sizes": {
            node.image.name: [int(node.image.size[0]), int(node.image.size[1])]
            for node in image_nodes
        },
    }
    return guide, vertices, faces, guide_regions, {
        "backend": args.guide_backend,
        "axis_contract": axis_contract,
        "import_cleanup": import_cleanup,
        "cleanup": cleanup,
        "pbr": pbr,
    }


def guide_polygon_uvs(guide):
    uv_layer = guide.data.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("Hunyuan guide has no active UV layer")
    values = []
    for polygon in guide.data.polygons:
        if polygon.loop_total != 3:
            raise RuntimeError("Hunyuan guide must remain triangulated")
        values.append(
            np.asarray(
                [tuple(uv_layer.data[index].uv) for index in polygon.loop_indices],
                dtype=np.float64,
            )
        )
    return values


def build_region_locked_source_uvs(
    template_vertices,
    template_regions,
    guide_vertices,
    guide_faces,
    guide_regions,
    guide,
    region_trees,
):
    guide_uv_layer = guide.data.uv_layers.active
    if guide_uv_layer is None:
        raise RuntimeError("Hunyuan guide UV layer is missing")
    polygon_uvs = guide_polygon_uvs(guide)
    source_uvs = np.zeros((len(template_vertices), 2), dtype=np.float64)
    face_indices = np.full(len(template_vertices), -1, dtype=np.int64)
    distances = np.zeros(len(template_vertices), dtype=np.float64)
    for index, (point, region) in enumerate(zip(template_vertices, template_regions)):
        nearest, _, face_index, distance = find_region_nearest(
            point, HumanRegion(int(region)), region_trees
        )
        triangle = guide_vertices[guide_faces[face_index]]
        barycentric = human_template_fit.triangle_barycentric_3d(
            nearest, triangle
        )
        source_uvs[index] = barycentric @ polygon_uvs[face_index]
        face_indices[index] = face_index
        distances[index] = distance
        if int(source_face_regions(guide_faces[[face_index]], guide_regions)[0]) != int(region):
            raise RuntimeError("texture correspondence crossed a human region")
    return {
        "source_uvs": source_uvs,
        "source_face_indices": face_indices,
        "distances": distances,
        "maximum_distance": float(distances.max(initial=0.0)),
    }


def image_pixels(image):
    width, height = map(int, image.size)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"image data is not loaded: {image.name}")
    values = np.asarray(image.pixels[:], dtype=np.float32)
    return values.reshape((height, width, 4))[::-1].copy()


def guide_pbr_images(guide):
    material = guide.material_slots[0].material
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError("appearance guide material has no node tree")
    by_name = {
        node.name: node.image
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE"
    }
    legacy_mapping = {
        "diffuse": by_name.get("Hunyuan Diffuse"),
        "metallic": by_name.get("Hunyuan Metallic"),
        "roughness": by_name.get("Hunyuan Roughness"),
    }
    if all(image is not None for image in legacy_mapping.values()):
        return {
            role: image_pixels(image) for role, image in legacy_mapping.items()
        }

    principled_nodes = [
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    ]
    if len(principled_nodes) != 1:
        raise RuntimeError("I23D guide must contain one Principled BSDF")
    principled = principled_nodes[0]
    base_links = principled.inputs["Base Color"].links
    metallic_links = principled.inputs["Metallic"].links
    roughness_links = principled.inputs["Roughness"].links
    if len(base_links) != 1 or len(metallic_links) != 1 or len(roughness_links) != 1:
        raise RuntimeError("I23D glTF PBR links are incomplete")
    base_node = base_links[0].from_node
    metallic_split = metallic_links[0].from_node
    roughness_split = roughness_links[0].from_node
    if base_node.type != "TEX_IMAGE" or base_node.image is None:
        raise RuntimeError("I23D Base Color must come from an image texture")
    if (
        metallic_split.type != "SEPARATE_COLOR"
        or roughness_split.type != "SEPARATE_COLOR"
        or metallic_links[0].from_socket.name != "Blue"
        or roughness_links[0].from_socket.name != "Green"
    ):
        raise RuntimeError("I23D metallic/roughness must use glTF Blue/Green channels")
    metallic_source_links = metallic_split.inputs["Color"].links
    roughness_source_links = roughness_split.inputs["Color"].links
    if len(metallic_source_links) != 1 or len(roughness_source_links) != 1:
        raise RuntimeError("I23D packed metallic/roughness image is missing")
    metallic_source = metallic_source_links[0].from_node
    roughness_source = roughness_source_links[0].from_node
    if (
        metallic_source.type != "TEX_IMAGE"
        or roughness_source.type != "TEX_IMAGE"
        or metallic_source.image is None
        or metallic_source.image != roughness_source.image
    ):
        raise RuntimeError("I23D metallic/roughness must share one packed image")

    packed = image_pixels(metallic_source.image)

    def channel_rgba(channel_index):
        rgb = np.repeat(packed[:, :, channel_index : channel_index + 1], 3, axis=2)
        return np.concatenate(
            (rgb, np.ones((*rgb.shape[:2], 1), dtype=rgb.dtype)), axis=2
        )

    return {
        "diffuse": image_pixels(base_node.image),
        "metallic": channel_rgba(2),
        "roughness": channel_rgba(1),
    }


def guide_region_palette(guide, source_uvs, template_regions):
    source_images = guide_pbr_images(guide)
    palette = {}
    expected_regions = tuple(int(region) for region in HumanRegion)
    for pbr_role, source_image in source_images.items():
        palette[pbr_role] = human_template_fit.region_palette_from_uv_samples(
            source_image,
            source_uvs,
            template_regions,
            expected_regions,
        )
    return palette


def rasterize_template_region_labels(
    source_mesh,
    template_regions,
    material_index,
    size,
):
    template_uv = source_mesh.data.uv_layers.active
    if template_uv is None:
        raise RuntimeError("Rocketbox template has no active UV layer")
    values = np.zeros((size, size, 1), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    for polygon in source_mesh.data.polygons:
        if polygon.material_index != material_index:
            continue
        polygon_regions = template_regions[
            np.asarray(polygon.vertices, dtype=np.int64)
        ]
        counts = np.bincount(polygon_regions, minlength=len(HumanRegion))
        region = int(np.argmax(counts))
        loops = list(polygon.loop_indices)
        for offset in range(1, len(loops) - 1):
            triangle_loops = (loops[0], loops[offset], loops[offset + 1])
            target_uv = np.asarray(
                [tuple(template_uv.data[index].uv) for index in triangle_loops],
                dtype=np.float64,
            )
            human_template_fit.rasterize_uv_triangle(
                values,
                mask,
                target_uv,
                np.zeros((3, 2), dtype=np.float64),
                np.asarray([[[float(region + 1)]]], dtype=np.float32),
                target_region=HumanRegion(region),
                source_region=HumanRegion(region),
            )
    labels = np.full((size, size), -1, dtype=np.int64)
    labels[mask] = np.rint(values[mask][:, 0]).astype(np.int64) - 1
    labels = human_template_fit.regularize_region_labels_by_island(labels, mask)
    return labels, mask


def resize_image_nearest(image, size):
    image = np.asarray(image)
    rows = np.rint(np.linspace(0, image.shape[0] - 1, size)).astype(np.int64)
    columns = np.rint(np.linspace(0, image.shape[1] - 1, size)).astype(np.int64)
    return image[rows[:, None], columns[None, :]].copy()


def official_color(asset_id, role):
    prefix = TEXTURE_PREFIXES[asset_id]
    path = source_review.require_texture(
        TEXTURE_DIRECTORIES[asset_id], f"{prefix}_{role}_color.tga"
    )
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    return image_pixels(image)


def write_numpy_image(name, values, path, non_color):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] not in (3, 4):
        raise RuntimeError("projected image must be HxWxRGB(A)")
    if values.shape[2] == 3:
        alpha = np.ones((*values.shape[:2], 1), dtype=np.float32)
        values = np.concatenate((values, alpha), axis=2)
    height, width = values.shape[:2]
    image = bpy.data.images.new(name=name, width=width, height=height, alpha=True)
    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    image.pixels = np.clip(values[::-1], 0.0, 1.0).reshape(-1).tolist()
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    image.pack()
    return image


def material_role_indices(mesh, texture_prefix):
    names = {
        slot.material.name: index
        for index, slot in enumerate(mesh.material_slots)
        if slot.material is not None
    }
    return {
        role: names[f"{texture_prefix}_{role}"]
        for role in ("body", "head", "opacity")
    }


def project_template_pbr(
    asset_id,
    source_mesh,
    source_uvs,
    guide,
    template_regions,
    output_dir,
):
    palette = guide_region_palette(guide, source_uvs, template_regions)
    material_indices = material_role_indices(source_mesh, TEXTURE_PREFIXES[asset_id])
    outputs = {}
    for material_index in (
        material_indices["body"],
        material_indices["head"],
    ):
        role = next(
            name for name in SURFACE_MATERIAL_ROLES
            if material_indices[name] == material_index
        )
        outputs[role] = {}
        labels, mask = rasterize_template_region_labels(
            source_mesh, template_regions, material_index, TEXTURE_SIZE
        )
        if not mask.any():
            raise RuntimeError(f"projected {role} PBR texture set is empty")
        original = resize_image_nearest(
            official_color(asset_id, role), TEXTURE_SIZE
        )
        diffuse_palette = {
            region: values[:3]
            for region, values in palette["diffuse"].items()
        }
        diffuse_rgb = human_template_fit.recolor_regions_preserve_luminance(
            original[:, :, :3], labels, diffuse_palette, strength=1.0
        )
        pbr_images = {
            "diffuse": np.concatenate(
                (
                    diffuse_rgb,
                    np.ones((*diffuse_rgb.shape[:2], 1), dtype=np.float64),
                ),
                axis=2,
            )
        }
        for pbr_role in ("metallic", "roughness"):
            image = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.float64)
            image[:, :, 3] = 1.0
            for region, color in palette[pbr_role].items():
                image[labels == int(region), :3] = color[:3]
            pbr_images[pbr_role] = human_template_fit.dilate_unpainted(
                image, mask.copy(), TEXTURE_DILATE_ITERATIONS
            )
        for pbr_role in PBR_ROLES:
            image = pbr_images[pbr_role]
            path = output_dir/f"{asset_id}_{role}_{pbr_role}.png"
            bpy_image = write_numpy_image(
                f"{asset_id}_{role}_{pbr_role}",
                image,
                path,
                pbr_role != "diffuse",
            )
            outputs[role][pbr_role] = {
                "image": bpy_image,
                "path": path,
                "painted_pixels": int(mask.sum()),
                "painted_fraction": float(mask.mean()),
            }
    outputs["opacity"] = {
        "source": "official_opacity_color",
        "alpha": "preserved",
    }
    return outputs


def install_surface_pbr(material, role, projected, normal_path):
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    diffuse = nodes.new("ShaderNodeTexImage")
    diffuse.name = f"Projected {role} Diffuse"
    diffuse.image = projected["diffuse"]["image"]
    metallic = nodes.new("ShaderNodeTexImage")
    metallic.name = f"Projected {role} Metallic"
    metallic.image = projected["metallic"]["image"]
    roughness = nodes.new("ShaderNodeTexImage")
    roughness.name = f"Projected {role} Roughness"
    roughness.image = projected["roughness"]["image"]
    links.new(diffuse.outputs["Color"], shader.inputs["Base Color"])
    links.new(metallic.outputs["Color"], shader.inputs["Metallic"])
    links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
    normal_path = source_review.require_texture(Path(normal_path).parent, Path(normal_path).name)
    normal_image = bpy.data.images.load(str(normal_path), check_existing=False)
    normal_image.colorspace_settings.name = "Non-Color"
    normal_image.pack()
    normal = nodes.new("ShaderNodeTexImage")
    normal.name = "official_normal"
    normal.image = normal_image
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])


def install_projected_materials(asset_id, armature, source_mesh, projected):
    prefix = TEXTURE_PREFIXES[asset_id]
    avatar = source_review.ImportedAvatar(
        mesh=source_mesh,
        armature=armature,
        imported_objects=(armature, source_mesh),
        material_slot_names=tuple(
            slot.material.name for slot in source_mesh.material_slots
        ),
    )
    provenance = source_review.reconnect_official_materials(
        avatar, TEXTURE_DIRECTORIES[asset_id], prefix
    )
    materials = {
        slot.material.name: slot.material for slot in source_mesh.material_slots
    }
    for role in SURFACE_MATERIAL_ROLES:
        normal_path = TEXTURE_DIRECTORIES[asset_id]/f"{prefix}_{role}_normal.tga"
        install_surface_pbr(
            materials[f"{prefix}_{role}"], role, projected[role], normal_path
        )
    opacity = materials[f"{prefix}_opacity"]
    opacity_nodes = [
        node for node in opacity.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.name == "official_opacity_color"
    ]
    if len(opacity_nodes) != 1:
        raise RuntimeError("official opacity alpha texture is missing")
    shaders = [
        node for node in opacity.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    ]
    if len(shaders) != 1 or not shaders[0].inputs["Alpha"].is_linked:
        raise RuntimeError("official opacity texture is not connected to Alpha")
    for material in materials.values():
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                if not node.image.has_data:
                    node.image.reload()
                    if len(node.image.pixels) <= 0:
                        raise RuntimeError(
                            f"material image has no pixels: {node.image.name}"
                        )
                    _ = node.image.pixels[0]
                node.image.pack()
    return provenance


def validate_projected_materials(source_mesh):
    result = {"material_slot_count": len(source_mesh.material_slots), "materials": {}}
    for slot in source_mesh.material_slots:
        material = slot.material
        images = []
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            if node.image.packed_file is None:
                raise RuntimeError(f"projected material image is not packed: {node.image.name}")
            if not node.image.has_data:
                if len(node.image.pixels) <= 0:
                    raise RuntimeError(
                        f"packed material image has no pixels: {node.image.name}"
                    )
                _ = node.image.pixels[0]
            if not node.image.has_data:
                raise RuntimeError(
                    f"packed material image did not decode: {node.image.name}"
                )
            images.append(
                {
                    "node": node.name,
                    "image": node.image.name,
                    "packed_size_bytes": int(node.image.packed_file.size),
                    "has_data": bool(node.image.has_data),
                }
            )
        if not images:
            raise RuntimeError(f"projected material has no images: {material.name}")
        result["materials"][material.name] = images
    return result


def remove_hy3d_guide(guide):
    mesh_data = guide.data
    bpy.data.objects.remove(guide, do_unlink=True)
    if mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def evaluated_mesh_world_points(mesh, indices=None):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        source = evaluated_mesh.vertices if indices is None else (
            evaluated_mesh.vertices[int(index)] for index in indices
        )
        return [evaluated.matrix_world @ vertex.co for vertex in source]
    finally:
        evaluated.to_mesh_clear()


def foot_weighted_vertex_indices(mesh):
    result = {}
    for side, names in FOOT_GROUPS.items():
        group_indices = {
            group.index for group in mesh.vertex_groups if group.name in names
        }
        indices = tuple(
            vertex.index
            for vertex in mesh.data.vertices
            if any(
                membership.group in group_indices and membership.weight > 0.0
                for membership in vertex.groups
            )
        )
        if not indices:
            raise RuntimeError(f"stable template has no {side} Foot/Toe vertices")
        result[side] = indices
    return result


def shift_action_object_z(action, offset, base_z, frame_start, frame_end):
    curves = [
        curve for curve in action.fcurves
        if curve.data_path == "location" and curve.array_index == 2
    ]
    if len(curves) > 1:
        raise RuntimeError("action has duplicate object Z location curves")
    if curves:
        curve = curves[0]
        for point in curve.keyframe_points:
            point.co.y += offset
            point.handle_left.y += offset
            point.handle_right.y += offset
    else:
        curve = action.fcurves.new(data_path="location", index=2)
        curve.keyframe_points.insert(frame_start, base_z + offset)
        curve.keyframe_points.insert(frame_end, base_z + offset)
    curve.update()


def sample_floor_state(armature, mesh, action, foot_indices):
    armature.animation_data.action = action
    frame_start, frame_end = direct.action_frame_range(action)
    minima = []
    side_minima = {side: [] for side in foot_indices}
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        points = evaluated_mesh_world_points(mesh)
        minima.append(min(float(point.z) for point in points))
        for side, indices in foot_indices.items():
            foot_points = evaluated_mesh_world_points(mesh, indices)
            side_minima[side].append(min(float(point.z) for point in foot_points))
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "mesh_minimum_z": min(minima),
        "side_minimum_z": {
            side: min(values) for side, values in side_minima.items()
        },
    }


def normalize_action_floor_contact(armature, mesh, action, floor_z_m):
    foot_indices = foot_weighted_vertex_indices(mesh)
    before = sample_floor_state(armature, mesh, action, foot_indices)
    minimum_foot_z = min(before["side_minimum_z"].values())
    offset = float(floor_z_m - minimum_foot_z)
    if abs(offset) > MAX_ACTION_FLOOR_OFFSET_M:
        raise RuntimeError(
            f"action floor offset {offset:.6f} m exceeds hard cap"
        )
    shift_action_object_z(
        action,
        offset,
        float(armature.location.z),
        before["frame_start"],
        before["frame_end"],
    )
    after = sample_floor_state(armature, mesh, action, foot_indices)
    penetration = max(0.0, floor_z_m - after["mesh_minimum_z"])
    if penetration > MAX_POST_NORMALIZE_PENETRATION_M:
        raise RuntimeError("normalized action still penetrates the fixed floor")
    support = {
        side: abs(value - floor_z_m)
        for side, value in after["side_minimum_z"].items()
    }
    for side in ("left", "right"):
        if support[side] > FOOT_SUPPORT_TOLERANCE_M:
            raise RuntimeError(f"normalized {side} foot never supports on fixed floor")
    return {
        "offset_m": offset,
        "before": before,
        "after": after,
        "maximum_penetration_m": penetration,
        "support_distance_m": support,
    }


def capture_action_skin_contract(armature, mesh, action):
    armature.animation_data.action = action
    frame_start, _ = direct.action_frame_range(action)
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    return direct.retarget.capture_skin_contract(mesh)


def texture_metrics_payload(projected):
    result = {}
    for role in SURFACE_MATERIAL_ROLES:
        result[role] = {}
        for pbr_role in PBR_ROLES:
            item = projected[role][pbr_role]
            result[role][pbr_role] = {
                "filename": item["path"].name,
                "sha256": direct.sha256_file(item["path"]),
                "size_bytes": item["path"].stat().st_size,
                "painted_pixels": item["painted_pixels"],
                "painted_fraction": item["painted_fraction"],
            }
    result["opacity"] = projected["opacity"]
    return result


def build_template_manifest(
    args,
    output_dir,
    action_metrics,
    source_hashes,
    current_hashes,
    floor_z_m,
    consumed_inputs,
    axis_contract,
    usage_scope,
    guide_provenance,
):
    return {
        "schema_version": "hy3d_rocketbox_bind_v1",
        "asset_id": args.asset_id,
        "binding_mode": BINDING_MODE,
        "usage_scope": USAGE_SCOPE
        if guide_provenance["backend"] == "hunyuan3d-2.1"
        else usage_scope,
        "guide_backend": guide_provenance["backend"],
        "research_release_ok": guide_provenance["research_release_ok"],
        "permissive_commercial_ok": guide_provenance[
            "permissive_commercial_ok"
        ],
        "guide_provenance": guide_provenance,
        "floor_z_m": floor_z_m,
        "reference": direct.file_descriptor(output_dir/"reference.png"),
        "glbs": {
            "walk": direct.file_descriptor(output_dir/"bound_walk.glb"),
            "idle": direct.file_descriptor(output_dir/"bound_idle.glb"),
        },
        "bound_blend": direct.file_descriptor(output_dir/"bound.blend"),
        "cleaned_obj_contract": {
            "role": "stable_template_geometry_only",
            "materials": False,
            "uv": True,
            "normals": True,
        },
        "action_names": {
            "walk": action_metrics["walk"]["action_name"],
            "idle": action_metrics["idle"]["action_name"],
        },
        "artifacts": {
            "cleaned_obj": direct.file_descriptor(output_dir/"cleaned.obj"),
            "bound_blend": direct.file_descriptor(output_dir/"bound.blend"),
            "bound_walk_glb": direct.file_descriptor(output_dir/"bound_walk.glb"),
            "bound_idle_glb": direct.file_descriptor(output_dir/"bound_idle.glb"),
            "bind_metrics": direct.file_descriptor(output_dir/"bind_metrics.json"),
            "template_fit_metrics": direct.file_descriptor(
                output_dir/"template_fit_metrics.json"
            ),
        },
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": consumed_inputs,
        "axis_contract": axis_contract,
    }


def build_stable_template(
    args,
    output_dir,
    baseline,
    guide_input,
    idle,
    source_hashes,
    snapshot,
):
    is_i23d = args.guide_glb is not None
    if is_i23d:
        usage_scope = i23d_rocketbox_contract.I23D_USAGE_SCOPE
        guide_provenance = guide_input["provenance"]
    else:
        usage_scope = USAGE_SCOPE
        guide_provenance = {
            "asset_id": args.asset_id,
            "backend": "hunyuan3d-2.1",
            "usage_scope": USAGE_SCOPE,
            "research_release_ok": False,
            "permissive_commercial_ok": False,
        }
    result = bpy.ops.wm.open_mainfile(
        filepath=str(snapshot["paths"]["baseline_blend"])
    )
    if "FINISHED" not in result:
        raise RuntimeError("could not open immutable Rocketbox baseline")
    direct.retarget.configure_animation_scene()
    armature, source_mesh = direct.identify_target_objects()
    runtime_mesh = source_mesh
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("approved Rocketbox walk action is missing")
    walk_action = armature.animation_data.action
    if walk_action.name != f"{args.asset_id}_walk_neutral_retarget":
        raise RuntimeError("approved walk_neutral action changed")
    walk_action.use_fake_user = True
    source = direct.capture_rocketbox_source(armature, source_mesh)
    floor_z_m = float(source["floor_z_m"])
    template_contract_before = capture_template_contract(source_mesh, armature)
    if is_i23d:
        guide, guide_vertices, guide_faces, guide_regions, guide_metrics = (
            import_clean_i23d_guide(
                snapshot["paths"]["i23d_guide_glb"], args, source
            )
        )
    else:
        pbr_paths = {
            role: snapshot["paths"][f"hy3d_{role}"]
            for role in direct.CONSUMED_HY3D_ROLES
        }
        guide, guide_vertices, guide_faces, guide_regions, guide_metrics = (
            import_clean_hy3d_guide(
                snapshot["paths"]["hy3d_paint_obj"],
                args.asset_id,
                source,
                pbr_paths,
            )
        )
    fit = fit_template_surface(
        source_mesh,
        armature,
        source,
        guide_vertices,
        guide_faces,
        guide_regions,
    )
    correspondence = build_region_locked_source_uvs(
        fit["template_vertices"],
        fit["template_regions"],
        guide_vertices,
        guide_faces,
        guide_regions,
        guide,
        fit["guide_region_trees"],
    )
    projected = project_template_pbr(
        args.asset_id,
        source_mesh,
        correspondence["source_uvs"],
        guide,
        fit["template_regions"],
        output_dir,
    )
    official_materials = install_projected_materials(
        args.asset_id, armature, source_mesh, projected
    )
    remove_hy3d_guide(guide)
    validate_template_contract(template_contract_before, source_mesh, armature)
    direct.validate_target_only_scene(armature, runtime_mesh)

    idle_action, idle_bake_metrics = direct.bake_idle_action(
        armature, args.asset_id, snapshot["paths"]["idle_motion_fbx"]
    )
    idle_action.use_fake_user = True
    action_set = direct.validate_two_actions(walk_action, idle_action)
    floor_normalization = {
        "walk": normalize_action_floor_contact(
            armature, source_mesh, walk_action, floor_z_m
        ),
        "idle": normalize_action_floor_contact(
            armature, source_mesh, idle_action, floor_z_m
        ),
    }
    action_metrics = {
        "walk": {
            "action_name": walk_action.name,
            "frame_start": direct.action_frame_range(walk_action)[0],
            "frame_end": direct.action_frame_range(walk_action)[1],
            "source": "approved baseline walk action with constant floor offset",
        },
        "idle": {
            "action_name": idle_action.name,
            "frame_start": direct.action_frame_range(idle_action)[0],
            "frame_end": direct.action_frame_range(idle_action)[1],
            "source": "gender-matched idle source-absolute bake with constant floor offset",
        },
    }
    cleaned_obj = direct.export_cleaned_obj(source_mesh, output_dir/"cleaned.obj")
    projected_validation = validate_projected_materials(source_mesh)
    direct.save_bound_blend(armature, source_mesh, output_dir/"bound.blend")

    walk_name = walk_action.name
    idle_name = idle_action.name
    armature, source_mesh, walk_action, idle_action = direct.load_saved_target(
        output_dir/"bound.blend", walk_name, idle_name
    )
    validate_template_contract(template_contract_before, source_mesh, armature)
    template_contract_after = capture_template_contract(source_mesh, armature)
    projected_validation = validate_projected_materials(source_mesh)
    expected_mesh = direct.retarget.mesh_metrics(source_mesh, armature)
    walk_expected_skin = capture_action_skin_contract(
        armature, source_mesh, walk_action
    )
    walk_start, walk_end, walk_positions = direct.sample_action_positions(
        armature, walk_action
    )
    idle_expected_skin = capture_action_skin_contract(
        armature, source_mesh, idle_action
    )
    idle_start, idle_end, idle_positions = direct.sample_action_positions(
        armature, idle_action
    )
    direct.isolate_action_for_export(armature, walk_action)
    direct.export_single_action_glb(
        armature, source_mesh, walk_action, output_dir/"bound_walk.glb"
    )
    armature, source_mesh, walk_action, idle_action = direct.load_saved_target(
        output_dir/"bound.blend", walk_name, idle_name
    )
    direct.isolate_action_for_export(armature, idle_action)
    direct.export_single_action_glb(
        armature, source_mesh, idle_action, output_dir/"bound_idle.glb"
    )
    glb_structure = {
        "walk": direct.inspect_bound_glb(output_dir/"bound_walk.glb"),
        "idle": direct.inspect_bound_glb(output_dir/"bound_idle.glb"),
    }
    reference_label = "i23d_reference" if is_i23d else "hy3d_reference"
    direct.atomic_copy(snapshot["paths"][reference_label], output_dir/"reference.png")
    glb_roundtrip = {
        "walk": direct.retarget.roundtrip_validate(
            output_dir/"bound_walk.glb",
            expected_mesh,
            walk_positions,
            walk_expected_skin,
            walk_start,
            walk_end,
        ),
        "idle": direct.retarget.roundtrip_validate(
            output_dir/"bound_idle.glb",
            expected_mesh,
            idle_positions,
            idle_expected_skin,
            idle_start,
            idle_end,
        ),
    }
    for role, values in glb_roundtrip.items():
        if not values["skin_weight_validation"]["passed"]:
            raise RuntimeError(f"{role} stable-template GLB skin validation failed")
        if values["maximum_world_joint_error_m"] >= values["joint_tolerance_m"]:
            raise RuntimeError(f"{role} stable-template GLB joint validation failed")

    if is_i23d:
        current_hashes = verify_i23d_source_hashes_current(
            baseline, guide_input, idle, source_hashes
        )
    else:
        current_hashes = direct.verify_source_hashes_current(
            baseline, guide_input, idle, source_hashes
        )
    fit_metrics = {
        "schema_version": (
            "i23d_rocketbox_template_fit_metrics_v1"
            if is_i23d
            else "hy3d_rocketbox_template_fit_metrics_v1"
        ),
        "asset_id": args.asset_id,
        "binding_mode": BINDING_MODE,
        "usage_scope": usage_scope,
        "guide_provenance": guide_provenance,
        "guide": guide_metrics,
        "surface_fit": fit["metrics"],
        "texture_correspondence": {
            "maximum_distance": correspondence["maximum_distance"],
            "matched_vertices": len(correspondence["source_uvs"]),
        },
        "projected_textures": texture_metrics_payload(projected),
        "official_materials": official_materials,
        "projected_material_validation": projected_validation,
        "template_contract_before": template_contract_before,
        "template_contract_after": template_contract_after,
        "floor_normalization": floor_normalization,
    }
    direct.atomic_write_json(output_dir/"template_fit_metrics.json", fit_metrics)
    bind_metrics = {
        "schema_version": (
            "i23d_rocketbox_bind_metrics_v1"
            if is_i23d
            else "hy3d_rocketbox_bind_metrics_v1"
        ),
        "asset_id": args.asset_id,
        "binding_mode": BINDING_MODE,
        "usage_scope": usage_scope,
        "guide_provenance": guide_provenance,
        "floor_z_m": floor_z_m,
        "axis_contract": guide_metrics["axis_contract"],
        "source_capture": {
            "mesh_name": source["mesh_name"],
            "vertex_count": source["vertex_count"],
            "face_count": source["face_count"],
            "bone_count": source["bone_count"],
            "uv_layer_count": source["uv_layer_count"],
            "material_slot_count": source["material_slot_count"],
        },
        "template_fit": fit_metrics,
        "cleaned_obj": cleaned_obj,
        "actions": action_metrics,
        "bound_action_set": action_set,
        "idle_bake": idle_bake_metrics,
        "glb_structure": glb_structure,
        "glb_roundtrip": glb_roundtrip,
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": snapshot["records"],
        "outputs": list(OUTPUT_FILENAMES),
    }
    direct.atomic_write_json(output_dir/"bind_metrics.json", bind_metrics)
    manifest = build_template_manifest(
        args,
        output_dir,
        action_metrics,
        source_hashes,
        current_hashes,
        floor_z_m,
        snapshot["records"],
        guide_metrics["axis_contract"],
        usage_scope,
        guide_provenance,
    )
    direct.atomic_write_json(output_dir/"bind_manifest.json", manifest)
    print(
        "ROCKETBOX_TEMPLATE_FIT_OK "
        f"asset_id={args.asset_id} backend={guide_provenance['backend']}"
    )
    return manifest


def run_template_fit(args):
    output_dir = direct.require_real_directory(args.output_dir, "output directory")
    baseline = direct.validate_baseline_inputs(args)
    idle = direct.validate_idle_motion(args)
    snapshot = None
    try:
        if args.guide_glb is not None:
            guide_input = validate_i23d_inputs(args)
            source_hashes = capture_i23d_source_hashes(
                baseline, guide_input, idle
            )
            snapshot = stage_i23d_input_snapshot(
                output_dir, baseline, guide_input, idle
            )
        else:
            guide_input = direct.validate_hy3d_inputs(args)
            source_hashes = direct.capture_source_hashes(
                baseline, guide_input, idle
            )
            snapshot = direct.stage_input_snapshot(
                output_dir, baseline, guide_input, idle
            )
        return build_stable_template(
            args,
            output_dir,
            baseline,
            guide_input,
            idle,
            source_hashes,
            snapshot,
        )
    finally:
        direct.cleanup_input_snapshot(snapshot)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direct.invalidate_readiness(args.output_dir)
    for filename in OUTPUT_FILENAMES:
        if filename in {"bind_manifest.json", "bind_metrics.json"}:
            continue
        try:
            (args.output_dir/filename).unlink()
        except FileNotFoundError:
            pass
    run_template_fit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
