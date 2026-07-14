"""Blender headless: robustly swap a textured mesh onto an existing dog rig.

This is an alternative to Blender's unconstrained Data Transfer modifier.  It
keeps the full target mesh, but transfers skin weights only from compatible
coarse dog regions (head->head, tail->tail, each leg->matching leg), then
inpaints any gaps along the target mesh graph.

Usage:
  blender --background --python tools/blender_robust_swap_mesh_keep_rig.py -- \\
    --rig-glb tmp/animated_dog/Dog_textured.glb.bak.warm_brown \\
    --new-mesh /data/jzy/code/Hunyuan3D-2.1/outputs/collie2_textured.obj \\
    --new-diffuse /tmp/collie2_diffuse_corrected.png \\
    --output tmp/hy3d/swap_test/Dog_robust_swap.glb \\
    --flip-x
"""
from __future__ import annotations

import argparse
import bmesh
import json
import math
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.robust_skin_transfer import (  # noqa: E402
    REGION_FRONT_LEFT_LEG,
    REGION_FRONT_RIGHT_LEG,
    REGION_HEAD,
    REGION_HIND_LEFT_LEG,
    REGION_HIND_RIGHT_LEG,
    REGION_NAMES,
    REGION_TAIL,
    REGION_TORSO,
    SkeletonCapsule,
    coarse_region_labels,
    face_region_labels,
    filter_gltf_animation_channels_json,
    graph_region_labels_from_capsules,
    ground_artifact_vertex_mask,
    inpaint_missing_weights,
    keep_top_k_normalized,
    low_limb_bridge_component_face_mask,
    low_limb_bridge_face_mask,
    mesh_bounds,
    normalize_rows,
    regularize_regions_by_connected_components,
    reverse_keyframe_time,
    target_region_labels_from_source_proximity,
    transfer_weights_by_nearest_surface,
    transfer_weights_by_region,
)


GLB_JSON_CHUNK = 0x4E4F534A


def postprocess_glb_animation_channels(
    path, keep_paths, *, canonical_walk_idle=False
):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"not a GLB file: {path}")

    offset = 12
    chunks = []
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks.append((chunk_type, data[offset:offset + chunk_length]))
        offset += chunk_length
    if not chunks or chunks[0][0] != GLB_JSON_CHUNK:
        raise ValueError(f"GLB missing JSON chunk: {path}")

    raw_json = chunks[0][1].rstrip(b" \t\r\n\x00")
    gltf = json.loads(raw_json.decode("utf-8"))
    removed = filter_gltf_animation_channels_json(gltf, keep_paths=keep_paths)
    changed = removed > 0
    if canonical_walk_idle:
        animations = list(gltf.get("animations", []))
        idle = next(
            (
                animation
                for animation in animations
                if str(animation.get("name", "")).lower().startswith("idle")
            ),
            None,
        )
        walking = next(
            (
                animation
                for animation in animations
                if str(animation.get("name", "")).lower() in {"walk", "walking"}
                or str(animation.get("name", "")).lower().startswith("walking_")
            ),
            None,
        )
        if idle is None or walking is None:
            raise ValueError(
                "exported GLB lacks canonical Idle and Walk/Walking animations: "
                f"{[animation.get('name') for animation in animations]}"
            )
        idle["name"] = "Idle"
        walking["name"] = "Walking"
        gltf["animations"] = [idle, walking]
        changed = True
    if not changed:
        return {
            "removed_channels": 0,
            "animation_names": [
                animation.get("name") for animation in gltf.get("animations", [])
            ],
        }

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)
    chunks[0] = (GLB_JSON_CHUNK, new_json)

    total_length = 12 + sum(8 + len(chunk) for _, chunk in chunks)
    out = bytearray(struct.pack("<4sII", b"glTF", version, total_length))
    for chunk_type, chunk in chunks:
        out += struct.pack("<II", len(chunk), chunk_type)
        out += chunk
    with open(path, "wb") as f:
        f.write(out)
    return {
        "removed_channels": removed,
        "animation_names": [
            animation.get("name") for animation in gltf.get("animations", [])
        ],
    }


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--rig-glb", required=True,
                   help="Original animated GLB/FBX providing armature, animations, and source weights.")
    p.add_argument("--new-mesh", required=True,
                   help="Target textured mesh (.obj/.glb/.gltf).")
    p.add_argument("--new-diffuse", default="",
                   help="Optional diffuse texture to assign to the target mesh.")
    p.add_argument("--output", required=True)
    p.add_argument("--auto-align", default="yes", choices=["yes", "no"])
    p.add_argument("--align-mode", default="uniform", choices=["uniform", "nonuniform"],
                   help="uniform preserves target proportions; nonuniform matches the rig bbox per axis.")
    p.add_argument("--flip-x", action="store_true",
                   help="Mirror the target along X before aligning.")
    p.add_argument("--target-rotate-z-deg", type=float, default=0.0,
                   help="Rotate the target mesh around Blender Z before bbox alignment and weight transfer.")
    p.add_argument(
        "--semantic-forward-axis",
        default="positive-x",
        choices=["positive-x", "negative-y"],
        help="Source-rig longitudinal axis used by anatomical region transfer. "
             "AnimalPack Cat/Dog use positive-x; Quaternius Farm Horse uses negative-y.",
    )
    p.add_argument("--max-distance-ratio", type=float, default=0.35,
                   help="Reject source matches farther than this fraction of the source bbox diagonal. "
                        "Use <=0 to disable distance rejection.")
    p.add_argument("--candidate-count", type=int, default=24,
                   help="Number of nearest compatible source face centers to evaluate per target vertex.")
    p.add_argument("--top-k", type=int, default=4,
                   help="Maximum bone influences to keep per target vertex.")
    p.add_argument("--min-weight", type=float, default=1e-5,
                   help="Do not write vertex-group weights below this value.")
    p.add_argument("--weight-mode", default="region", choices=["region", "auto", "nearest"],
                   help="region copies compatible source weights; auto uses Blender automatic weights "
                        "against the original armature after alignment; nearest copies from nearest "
                        "source mesh surface without animal semantic regions.")
    p.add_argument(
        "--nearest-backend",
        default="bvh",
        choices=["bvh", "bruteforce"],
        help=(
            "Nearest-surface query backend. bvh uses Blender's C-level triangle "
            "BVH and is the production default; bruteforce preserves the older "
            "reference implementation for small regression fixtures."
        ),
    )
    p.add_argument("--segmentation-mode", default="proximity",
                   choices=["proximity", "skeleton-graph", "proximity-components"],
                   help="Target anatomical segmentation mode used by --weight-mode region.")
    p.add_argument("--graph-unary-weight", type=float, default=1.5,
                   help="Skeleton graph segmentation preference for staying near matching bone capsules.")
    p.add_argument("--graph-seed-distance-ratio", type=float, default=0.05,
                   help="Seed vertices within this source-bbox diagonal ratio of a semantic bone capsule.")
    p.add_argument("--protected-region-unary-scale", type=float, default=0.25,
                   help="Lower capsule-distance penalty for connected head/tail graph propagation.")
    p.add_argument("--component-regularize-min-size", type=int, default=3,
                   help="Minimum connected shell size for proximity-components regularization.")
    p.add_argument("--dampen-foot-rotations", type=float, default=1.0,
                   help="Scale foot-end bone rotations toward identity in exported actions. "
                        "1.0 keeps source animation; 0.5 halves foot rotation.")
    p.add_argument("--foot-bones", default="Bone.010,Bone.013,Bone.016,Bone.019",
                   help="Comma-separated distal foot bones for --dampen-foot-rotations.")
    p.add_argument("--dampen-head-rotations", type=float, default=0.0,
                   help="Scale head/neck rotations toward identity. 0 freezes source head bob.")
    p.add_argument("--head-bones", default="Bone.002,Bone.003,Bone.003_end")
    p.add_argument("--dampen-tail-rotations", type=float, default=0.0,
                   help="Scale tail rotations toward identity. 0 keeps tail rigid relative to body.")
    p.add_argument("--tail-bones", default="Bone.004,Bone.005,Bone.006,Bone.007,Bone.007_end")
    p.add_argument("--reverse-actions", default="no", choices=["yes", "no"],
                   help="Reverse matching action keyframes in time, useful for backwards walk clips.")
    p.add_argument("--reverse-action-hints", default="Walking",
                   help="Comma-separated action name substrings used by --reverse-actions.")
    p.add_argument("--animation-export-mode", default="actions", choices=["actions", "nla"],
                   help="Use actions for GLB sources; use nla when importing multi-action FBX rigs.")
    p.add_argument(
        "--export-action-policy",
        default="all",
        choices=["all", "walk-idle"],
        help="Limit exported actions to one canonical Walking and one canonical Idle. "
             "Use walk-idle for farm rigs whose source also contains Death/Run/Jump/WalkSlow.",
    )
    p.add_argument("--ue-safe-animation-channels", default="yes", choices=["yes", "no"],
                   help="Post-process exported GLB animations for UE Interchange stability.")
    p.add_argument("--ue-animation-keep-paths", default="translation,rotation",
                   help="Comma-separated glTF animation target paths kept by --ue-safe-animation-channels.")
    p.add_argument("--backface-culling", default="no", choices=["yes", "no"],
                   help="Use single-sided material export. Usually keep no for Hunyuan fur shells.")
    p.add_argument("--remove-ground-artifacts", default="yes", choices=["yes", "no"],
                   help="Delete low, flat disconnected floor/shadow cards before skin transfer.")
    p.add_argument("--ground-artifact-max-center-height-ratio", type=float, default=0.035)
    p.add_argument("--ground-artifact-max-component-height-ratio", type=float, default=0.015)
    p.add_argument("--ground-artifact-min-horizontal-spread-ratio", type=float, default=0.12)
    p.add_argument("--ground-artifact-min-vertices", type=int, default=20)
    p.add_argument("--remove-limb-bridges", default="yes", choices=["yes", "no"],
                   help="Cut low cross-limb faces from the weight-transfer graph.")
    p.add_argument("--delete-limb-bridge-faces", default="no", choices=["yes", "no"],
                   help="Also remove low cross-limb bridge faces from the rendered mesh. "
                        "Keep no for generated meshes: the graph cut is useful for weight "
                        "propagation, but deleting those faces can open the belly and paws.")
    p.add_argument("--limb-bridge-max-center-height-ratio", type=float, default=0.35)
    p.add_argument("--remove-limb-bridge-components", default="yes", choices=["yes", "no"],
                   help="Remove small low mixed-limb components left after bridge-face cuts.")
    p.add_argument("--limb-bridge-component-min-direct-faces", type=int, default=200,
                   help="Only run component cleanup when the direct bridge-face cut is at least this large.")
    p.add_argument("--limb-bridge-component-max-center-height-ratio", type=float, default=0.42)
    p.add_argument("--limb-bridge-component-min-faces", type=int, default=2)
    p.add_argument("--limb-bridge-component-max-faces", type=int, default=700)
    p.add_argument("--limb-bridge-component-min-limb-regions", type=int, default=2)
    p.add_argument("--limb-bridge-component-min-limb-fraction", type=float, default=0.50)
    p.add_argument("--limb-bridge-component-max-anchor-fraction", type=float, default=0.40)
    return p.parse_args(argv)


def import_mesh_only(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        # Stable Rocketbox animal templates are distributed as skinned FBX.
        # Only the imported mesh is returned; robust_transfer replaces its
        # original parenting and weights with the selected runtime rig.
        bpy.ops.import_scene.fbx(filepath=path, use_anim=False)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh format {ext}")
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def import_rig_scene(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise SystemExit(f"unsupported rig format {ext}")


def keep_canonical_walk_idle_actions(armature):
    actions = list(bpy.data.actions)
    base_name = lambda action: action.name.lower().split("_", 1)[0]
    idle = next((action for action in actions if base_name(action) == "idle"), None)
    walking = next(
        (
            action
            for action in actions
            if base_name(action) in {"walk", "walking"}
        ),
        None,
    )
    if idle is None or walking is None:
        raise SystemExit(
            "walk-idle export requires exact Idle and Walk/Walking actions; "
            f"available={[action.name for action in actions]}"
        )
    idle.name = "Idle"
    walking.name = "Walking"
    for action in actions:
        if action not in {idle, walking}:
            bpy.data.actions.remove(action)
    armature.animation_data_create()
    armature.animation_data.action = idle
    print("[anim] canonical export actions=['Idle', 'Walking']", flush=True)


def object_bbox_world(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def align_bbox(target_obj, src_obj, mode):
    smn, smx = object_bbox_world(src_obj)
    tmn, tmx = object_bbox_world(target_obj)
    src_ext = smx - smn
    tgt_ext = tmx - tmn
    if mode == "nonuniform":
        scale = (
            src_ext.x / max(tgt_ext.x, 1e-12),
            src_ext.y / max(tgt_ext.y, 1e-12),
            src_ext.z / max(tgt_ext.z, 1e-12),
        )
    else:
        factor = max(src_ext) / max(tgt_ext)
        scale = (factor, factor, factor)
    target_obj.scale = (
        target_obj.scale.x * scale[0],
        target_obj.scale.y * scale[1],
        target_obj.scale.z * scale[2],
    )
    bpy.context.view_layer.update()

    tmn, tmx = object_bbox_world(target_obj)
    delta = ((smn + smx) * 0.5) - ((tmn + tmx) * 0.5)
    target_obj.location += delta
    bpy.context.view_layer.update()
    print(f"[align] mode={mode} src_ext={list(src_ext)} tgt_ext={list(tgt_ext)} "
          f"scale={scale} translate={list(delta)}", flush=True)


def apply_transforms(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def rotate_target_z_degrees(obj, degrees):
    if abs(float(degrees)) <= 1e-9:
        return
    rotation = Matrix.Rotation(math.radians(float(degrees)), 4, "Z")
    obj.data.transform(rotation)
    obj.data.update()
    bpy.context.view_layer.update()
    print(f"[target-rotate] rotated target around Z by {float(degrees):.3f} deg", flush=True)


def recompute_normals(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def weld_target_position_duplicates(obj):
    """Share geometry vertices split by glTF UV/normal seams before skinning.

    Blender's glTF importer creates separate vertices at UV and normal seams.
    If weights are transferred to those copies independently, coincident copies
    can receive different bone weights and fan apart during animation.  UV
    coordinates remain stored per face corner in Blender, so welding geometry
    vertices does not collapse the texture layout.
    """
    before_vertices = len(obj.data.vertices)
    before_faces = len(obj.data.polygons)
    if before_vertices == 0:
        raise SystemExit("target mesh has no vertices before position weld")
    coordinates = [vertex.co.copy() for vertex in obj.data.vertices]
    extent = max(
        max(point[axis] for point in coordinates)
        - min(point[axis] for point in coordinates)
        for axis in range(3)
    )
    distance = max(float(extent) * 1.0e-7, 1.0e-9)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
        bm.to_mesh(obj.data)
    finally:
        bm.free()
    obj.data.update(calc_edges=True)
    after_vertices = len(obj.data.vertices)
    after_faces = len(obj.data.polygons)
    if after_faces != before_faces:
        raise SystemExit(
            "target position weld changed face count: "
            f"{before_faces} -> {after_faces}"
        )
    print(
        "[target-weld] shared glTF seam geometry before skin transfer "
        f"vertices={before_vertices}->{after_vertices} "
        f"welded={before_vertices - after_vertices} distance={distance:.9g} "
        f"faces={before_faces}",
        flush=True,
    )
    return {
        "vertices_before": before_vertices,
        "vertices_after": after_vertices,
        "vertices_welded": before_vertices - after_vertices,
        "distance": distance,
        "faces": before_faces,
    }


def mesh_vertices_world(obj):
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)


def mesh_vertices_local(obj):
    return np.array([v.co for v in obj.data.vertices], dtype=np.float64)


def dog_region_coords(vertices, forward_axis="positive-x"):
    """Convert Blender coordinates to helper convention: front, up, left.

    Blender is Z-up here.  The pure helper module uses X=front, Y=up, Z=side.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if forward_axis == "positive-x":
        return vertices[:, [0, 2, 1]]
    if forward_axis == "negative-y":
        return np.column_stack((-vertices[:, 1], vertices[:, 2], vertices[:, 0]))
    raise ValueError(f"unsupported semantic forward axis: {forward_axis}")


def _semantic_region_from_point(point_world, region_bounds, forward_axis):
    point = dog_region_coords(
        np.asarray([point_world], dtype=np.float64), forward_axis
    )[0]
    mn, mx = region_bounds
    uvw = (point - mn) / np.maximum(mx - mn, 1e-12)
    x, y, z = uvw
    if x <= 0.22 and y >= 0.36:
        return REGION_TAIL
    if x >= 0.64 and y >= 0.36:
        return REGION_HEAD
    if y < 0.44 and x > 0.16:
        front = x >= 0.50
        left = z >= 0.50
        if front and left:
            return REGION_FRONT_LEFT_LEG
        if front:
            return REGION_FRONT_RIGHT_LEG
        if left:
            return REGION_HIND_LEFT_LEG
        return REGION_HIND_RIGHT_LEG
    return REGION_TORSO


def armature_region_capsules(
    armature_obj, region_bounds, source_world_diag, forward_axis="positive-x"
):
    radius_by_region = {
        REGION_TAIL: 0.025,
        REGION_TORSO: 0.060,
        REGION_HEAD: 0.045,
        REGION_FRONT_LEFT_LEG: 0.035,
        REGION_FRONT_RIGHT_LEG: 0.035,
        REGION_HIND_LEFT_LEG: 0.035,
        REGION_HIND_RIGHT_LEG: 0.035,
    }
    capsules = []
    counts = {}
    for bone in armature_obj.data.bones:
        if bone.name.lower().startswith("ik"):
            continue
        head = np.asarray(armature_obj.matrix_world @ bone.head_local, dtype=np.float64)
        tail = np.asarray(armature_obj.matrix_world @ bone.tail_local, dtype=np.float64)
        length = float(np.linalg.norm(tail - head))
        if length <= 1e-8:
            continue
        region = _semantic_region_from_point(
            (head + tail) * 0.5, region_bounds, forward_axis
        )
        radius = max(float(source_world_diag) * radius_by_region[region], length * 0.18)
        capsules.append(SkeletonCapsule(region=region, start=head, end=tail, radius=radius))
        counts[REGION_NAMES.get(region, str(region))] = counts.get(REGION_NAMES.get(region, str(region)), 0) + 1
    print(f"[capsules] count={len(capsules)} regions={counts}", flush=True)
    return capsules


def triangulated_faces_with_polygon_indices(obj):
    faces = []
    polygon_indices = []
    for poly in obj.data.polygons:
        verts = list(poly.vertices)
        if len(verts) < 3:
            continue
        for i in range(1, len(verts) - 1):
            faces.append((verts[0], verts[i], verts[i + 1]))
            polygon_indices.append(poly.index)
    if not faces:
        raise SystemExit(f"mesh {obj.name} has no faces")
    return np.asarray(faces, dtype=np.int64), np.asarray(polygon_indices, dtype=np.int64)


def triangulated_faces(obj):
    return triangulated_faces_with_polygon_indices(obj)[0]


def source_vertex_weights(src_obj):
    group_names = [vg.name for vg in src_obj.vertex_groups]
    weights = np.zeros((len(src_obj.data.vertices), len(group_names)), dtype=np.float64)
    for vertex in src_obj.data.vertices:
        for group in vertex.groups:
            if group.group < weights.shape[1]:
                weights[vertex.index, group.group] = group.weight
    if not group_names:
        raise SystemExit(f"source mesh {src_obj.name} has no vertex groups")
    if np.count_nonzero(weights.sum(axis=1) > 1e-8) == 0:
        raise SystemExit(f"source mesh {src_obj.name} vertex groups are empty")
    return group_names, weights


def clear_vertex_groups(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    while obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups[0])


def write_vertex_groups(obj, group_names, weights, min_weight):
    clear_vertex_groups(obj)
    groups = [obj.vertex_groups.new(name=name) for name in group_names]
    written = 0
    zero_rows = 0
    for vertex_index, row in enumerate(weights):
        nz = np.flatnonzero(row > min_weight)
        if len(nz) == 0:
            zero_rows += 1
            continue
        for group_index in nz:
            groups[int(group_index)].add([vertex_index], float(row[group_index]), "ADD")
            written += 1
    print(f"[weights] wrote {written} assignments across {len(groups)} groups "
          f"(zero_rows={zero_rows})", flush=True)


def delete_vertices_by_mask(obj, mask, reason):
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        print(f"[cleanup] {reason}: no vertices removed", flush=True)
        return 0
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    for poly in obj.data.polygons:
        poly.select = False
    for edge in obj.data.edges:
        edge.select = False
    for vertex in obj.data.vertices:
        vertex.select = bool(mask[vertex.index])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update()
    bpy.context.view_layer.update()
    print(f"[cleanup] {reason}: removed {count} vertices", flush=True)
    return count


def delete_faces_by_mask(obj, mask, polygon_indices, reason):
    mask = np.asarray(mask, dtype=bool)
    polygon_indices = np.asarray(polygon_indices, dtype=np.int64)
    if len(mask) != len(polygon_indices):
        raise ValueError("face mask and polygon index arrays must have the same length")
    selected_polygons = {int(poly_index) for poly_index in polygon_indices[mask]}
    count = len(selected_polygons)
    if count == 0:
        print(f"[cleanup] {reason}: no faces removed", flush=True)
        return 0
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    selected_edges = set()
    for vertex in obj.data.vertices:
        vertex.select = False
    for edge in obj.data.edges:
        edge.select = False
    for poly in obj.data.polygons:
        selected = poly.index in selected_polygons
        poly.select = selected
        if selected:
            selected_edges.update(poly.edge_keys)
    edge_keys = {tuple(sorted(edge.vertices)): edge.index for edge in obj.data.edges}
    for edge_key in selected_edges:
        edge_index = edge_keys.get(tuple(sorted(edge_key)))
        if edge_index is not None:
            obj.data.edges[edge_index].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="EDGE_FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update()
    bpy.context.view_layer.update()
    print(f"[cleanup] {reason}: removed {count} faces and selected boundary edges", flush=True)
    return count


def remove_ground_artifacts(mesh_obj, args):
    if args.remove_ground_artifacts != "yes":
        print("[cleanup] ground artifacts disabled", flush=True)
        return 0
    apply_transforms(mesh_obj)
    vertices = mesh_vertices_local(mesh_obj)
    faces = triangulated_faces(mesh_obj)
    artifact = ground_artifact_vertex_mask(
        vertices=vertices,
        faces=faces,
        up_axis=2,
        max_center_height_ratio=args.ground_artifact_max_center_height_ratio,
        max_component_height_ratio=args.ground_artifact_max_component_height_ratio,
        min_horizontal_spread_ratio=args.ground_artifact_min_horizontal_spread_ratio,
        min_vertices=args.ground_artifact_min_vertices,
    )
    return delete_vertices_by_mask(mesh_obj, artifact, "ground artifacts")


def filter_limb_bridge_faces(vertices, faces, target_regions, args, mesh_obj=None, polygon_indices=None):
    if args.remove_limb_bridges != "yes":
        print("[cleanup] limb bridge graph cut disabled", flush=True)
        return faces, 0
    bridge = low_limb_bridge_face_mask(
        vertices=vertices,
        faces=faces,
        vertex_regions=target_regions,
        up_axis=2,
        max_center_height_ratio=args.limb_bridge_max_center_height_ratio,
    )
    face_count = int(bridge.sum())
    component_count = 0
    if (
        args.remove_limb_bridge_components == "yes"
        and face_count >= int(args.limb_bridge_component_min_direct_faces)
    ):
        keep_indices = np.flatnonzero(~bridge)
        if len(keep_indices):
            component_bridge = low_limb_bridge_component_face_mask(
                vertices=vertices,
                faces=faces[keep_indices],
                vertex_regions=target_regions,
                up_axis=2,
                max_center_height_ratio=args.limb_bridge_component_max_center_height_ratio,
                min_component_faces=args.limb_bridge_component_min_faces,
                max_component_faces=args.limb_bridge_component_max_faces,
                min_limb_regions=args.limb_bridge_component_min_limb_regions,
                min_limb_vertex_fraction=args.limb_bridge_component_min_limb_fraction,
                max_anchor_vertex_fraction=args.limb_bridge_component_max_anchor_fraction,
            )
            bridge[keep_indices[component_bridge]] = True
            component_count = int(component_bridge.sum())
    elif args.remove_limb_bridge_components == "yes":
        print(f"[cleanup] limb bridge component cut skipped: direct={face_count} "
              f"< min_direct={args.limb_bridge_component_min_direct_faces}", flush=True)
    else:
        print("[cleanup] limb bridge component cut disabled", flush=True)
    count = int(bridge.sum())
    if count == 0:
        print("[cleanup] limb bridge graph cut: no faces cut", flush=True)
        return faces, 0
    print(f"[cleanup] limb bridge graph cut: cut {count} faces from weight graph "
          f"(direct={face_count}, components={component_count})", flush=True)
    if args.delete_limb_bridge_faces == "yes":
        if mesh_obj is None or polygon_indices is None:
            raise ValueError("mesh_obj and polygon_indices are required to delete bridge geometry")
        delete_faces_by_mask(mesh_obj, bridge, polygon_indices, "limb bridge geometry")
    return faces[~bridge], count


def assign_texture(mesh_obj, diffuse_path, use_backface_culling=False):
    mat = bpy.data.materials.new(name="RobustSwap_Fur")
    mat.use_nodes = True
    mat.use_backface_culling = bool(use_backface_culling)
    mat.show_transparent_back = not bool(use_backface_culling)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(diffuse_path)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    print(f"[texture] assigned {diffuse_path} backface_culling={use_backface_culling}", flush=True)


def parent_to_armature(mesh_obj, armature_obj):
    mesh_obj.parent = armature_obj
    mesh_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = armature_obj
    mod.use_vertex_groups = True
    print(f"[parent] {mesh_obj.name} -> {armature_obj.name}", flush=True)


def auto_weight_to_armature(mesh_obj, armature_obj):
    apply_transforms(mesh_obj)
    clear_vertex_groups(mesh_obj)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    print(f"[auto-weight] bound {mesh_obj.name} to {armature_obj.name} with Blender automatic weights",
          flush=True)


def summarize_region_weights(weights, labels, group_names, title):
    print(f"[diag] {title}", flush=True)
    for region in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == region)
        if len(idx) == 0:
            continue
        mass = weights[idx].sum(axis=0)
        if mass.sum() <= 1e-12:
            print(f"  {REGION_NAMES.get(region, region)} count={len(idx)} no weight", flush=True)
            continue
        order = np.argsort(mass)[-5:][::-1]
        top = ", ".join(f"{group_names[j]}={mass[j] / mass.sum():.2f}" for j in order if mass[j] > 0)
        print(f"  {REGION_NAMES.get(region, region)} count={len(idx)} top={top}", flush=True)


def dampen_bone_rotation_actions(bone_names, factor):
    if abs(factor - 1.0) < 1e-9:
        return
    factor = max(0.0, min(1.0, float(factor)))
    identity = Quaternion((1.0, 0.0, 0.0, 0.0))
    changed = 0
    for action in bpy.data.actions:
        for bone_name in bone_names:
            path = f'pose.bones["{bone_name}"].rotation_quaternion'
            curves = {fc.array_index: fc for fc in action.fcurves if fc.data_path == path}
            if len(curves) != 4:
                continue
            frames = sorted({round(k.co.x, 6) for fc in curves.values() for k in fc.keyframe_points})
            key_by_frame = {
                axis: {round(k.co.x, 6): k for k in fc.keyframe_points}
                for axis, fc in curves.items()
            }
            for frame in frames:
                q = Quaternion([curves[axis].evaluate(frame) for axis in range(4)])
                q.normalize()
                q2 = identity.slerp(q, factor)
                for axis in range(4):
                    key = key_by_frame[axis].get(frame)
                    if key is None:
                        continue
                    key.co.y = q2[axis]
                    key.handle_left.y = q2[axis]
                    key.handle_right.y = q2[axis]
                    key.interpolation = "LINEAR"
                    changed += 1
            for fc in curves.values():
                fc.update()
    print(f"[anim] dampened rotations for bones={bone_names} factor={factor} "
          f"changed_keys={changed}", flush=True)


def reverse_matching_actions(action_hints):
    hints = [hint.strip().lower() for hint in action_hints.split(",") if hint.strip()]
    if not hints:
        print("[anim] reverse requested without action hints; skipped", flush=True)
        return
    changed = 0
    reversed_actions = []
    for action in bpy.data.actions:
        if not any(hint in action.name.lower() for hint in hints):
            continue
        start, end = action.frame_range
        for fc in action.fcurves:
            for key in fc.keyframe_points:
                frame, left, right = reverse_keyframe_time(
                    frame=key.co.x,
                    handle_left=key.handle_left.x,
                    handle_right=key.handle_right.x,
                    start=start,
                    end=end,
                )
                key.co.x = frame
                key.handle_left.x = left
                key.handle_right.x = right
                changed += 1
            fc.update()
        reversed_actions.append(action.name)
    print(f"[anim] reversed actions={reversed_actions} changed_keys={changed}", flush=True)


def stash_armature_actions_for_gltf(armature_obj):
    armature_obj.animation_data_create()
    while armature_obj.animation_data.nla_tracks:
        armature_obj.animation_data.nla_tracks.remove(armature_obj.animation_data.nla_tracks[0])
    action_names = []
    bone_prefix = 'pose.bones["'
    for action in bpy.data.actions:
        if not any(fc.data_path.startswith(bone_prefix) for fc in action.fcurves):
            continue
        start, end = action.frame_range
        track = armature_obj.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, int(round(start)), action)
        strip.name = action.name
        strip.action_frame_start = start
        strip.action_frame_end = end
        action_names.append(action.name)
    print(f"[anim] stashed actions for glTF export: {action_names}", flush=True)


def robust_transfer(src_mesh, tgt_mesh, armature_obj, args):
    src_vertices = mesh_vertices_world(src_mesh)
    src_faces = triangulated_faces(src_mesh)
    group_names, src_weights = source_vertex_weights(src_mesh)
    src_region_vertices = dog_region_coords(
        src_vertices, args.semantic_forward_axis
    )
    bounds = mesh_bounds(src_region_vertices)
    source_face_regions = face_region_labels(src_region_vertices, src_faces, bounds=bounds)

    apply_transforms(tgt_mesh)
    tgt_vertices = mesh_vertices_local(tgt_mesh)
    tgt_faces, tgt_polygon_indices = triangulated_faces_with_polygon_indices(tgt_mesh)
    coarse_target_regions = coarse_region_labels(
        dog_region_coords(tgt_vertices, args.semantic_forward_axis), bounds=bounds
    )
    weight_graph_faces = tgt_faces

    diag = float(np.linalg.norm(mesh_bounds(src_vertices)[1] - mesh_bounds(src_vertices)[0]))
    max_distance = None if args.max_distance_ratio <= 0 else diag * args.max_distance_ratio
    if args.segmentation_mode == "skeleton-graph":
        capsules = armature_region_capsules(
            armature_obj, bounds, diag, args.semantic_forward_axis
        )
        target_regions, graph_stats = graph_region_labels_from_capsules(
            vertices=tgt_vertices,
            faces=tgt_faces,
            capsules=capsules,
            coarse_labels=coarse_target_regions,
            unary_weight=args.graph_unary_weight,
            protected_region_unary_scale=args.protected_region_unary_scale,
            seed_distance_ratio=args.graph_seed_distance_ratio,
        )
        print(f"[segmentation] mode=skeleton-graph stats={graph_stats}", flush=True)
    else:
        target_regions = target_region_labels_from_source_proximity(
            source_vertices=src_vertices,
            source_faces=src_faces,
            source_face_regions=source_face_regions,
            target_vertices=tgt_vertices,
            coarse_target_regions=coarse_target_regions,
        )
        weight_graph_faces, _ = filter_limb_bridge_faces(
            tgt_vertices, tgt_faces, target_regions, args,
            mesh_obj=tgt_mesh, polygon_indices=tgt_polygon_indices,
        )
        if args.segmentation_mode == "proximity-components":
            target_regions, component_stats = regularize_regions_by_connected_components(
                faces=weight_graph_faces,
                labels=target_regions,
                eligible_regions={
                    REGION_TAIL,
                    REGION_TORSO,
                    REGION_HIND_LEFT_LEG,
                    REGION_HIND_RIGHT_LEG,
                },
                vote_bias={
                    REGION_TAIL: 1.15,
                    REGION_TORSO: 1.20,
                    REGION_HIND_LEFT_LEG: 0.90,
                    REGION_HIND_RIGHT_LEG: 0.90,
                },
                min_component_size=args.component_regularize_min_size,
            )
            print(f"[segmentation] mode=proximity-components stats={component_stats}", flush=True)
        else:
            print("[segmentation] mode=proximity", flush=True)
    if args.segmentation_mode == "skeleton-graph":
        weight_graph_faces, _ = filter_limb_bridge_faces(
            tgt_vertices, tgt_faces, target_regions, args,
            mesh_obj=tgt_mesh, polygon_indices=tgt_polygon_indices,
        )
    print(f"[transfer] source verts={len(src_vertices)} faces={len(src_faces)} "
          f"target verts={len(tgt_vertices)} faces={len(tgt_faces)} "
          f"weight_graph_faces={len(weight_graph_faces)} groups={len(group_names)}", flush=True)
    print(f"[transfer] bbox_diag={diag:.4f} max_distance={max_distance}", flush=True)

    transferred, matched, stats = transfer_weights_by_region(
        source_vertices=src_vertices,
        source_faces=src_faces,
        source_weights=src_weights,
        target_vertices=tgt_vertices,
        source_face_regions=source_face_regions,
        target_regions=target_regions,
        max_distance=max_distance,
        candidate_count=args.candidate_count,
    )
    print(f"[transfer] stats={stats}", flush=True)

    filled_weights, filled_mask, n_iters = inpaint_missing_weights(
        weight_graph_faces,
        transferred,
        matched,
        max_iterations=96,
    )
    print(f"[inpaint] filled={int(filled_mask.sum())}/{len(filled_mask)} iterations={n_iters}", flush=True)
    if not bool(np.all(filled_mask)):
        raise SystemExit(
            "skin transfer left unweighted target vertices: "
            f"{int((~filled_mask).sum())}/{len(filled_mask)}"
        )

    final_weights = keep_top_k_normalized(filled_weights, k=args.top_k)
    summarize_region_weights(final_weights, target_regions, group_names, "target region weight mass")
    write_vertex_groups(tgt_mesh, group_names, final_weights, args.min_weight)


def transfer_weights_by_blender_bvh(
    source_vertices,
    source_faces,
    source_weights,
    target_vertices,
    *,
    max_distance=None,
):
    """Interpolate weights from the exact nearest source triangle via Blender BVH.

    The legacy NumPy reference implementation scans every source face center
    for every target vertex.  That is useful for tiny unit fixtures but becomes
    quadratic at runtime-mesh scale.  Blender's BVH keeps the same
    nearest-surface/barycentric contract while moving spatial lookup into its
    C implementation.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    source_weights = normalize_rows(source_weights)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)

    if len(source_faces) == 0:
        return (
            np.zeros((len(target_vertices), source_weights.shape[1]), dtype=np.float64),
            np.zeros(len(target_vertices), dtype=bool),
            {
                "backend": "blender_bvh",
                "target_vertices": int(len(target_vertices)),
                "matched": 0,
                "unmatched": int(len(target_vertices)),
                "no_candidates": int(len(target_vertices)),
                "over_distance": 0,
            },
        )

    bvh = BVHTree.FromPolygons(
        [tuple(float(x) for x in row) for row in source_vertices],
        [tuple(int(x) for x in row) for row in source_faces],
        all_triangles=True,
    )
    face_vertices = source_vertices[source_faces]
    out = np.zeros((len(target_vertices), source_weights.shape[1]), dtype=np.float64)
    matched = np.zeros(len(target_vertices), dtype=bool)
    no_candidates = 0
    over_distance = 0

    for i, point in enumerate(target_vertices):
        hit = bvh.find_nearest(Vector(point))
        if hit is None or hit[0] is None or hit[2] is None:
            no_candidates += 1
            continue
        location, _normal, face_index, distance = hit
        if max_distance is not None and float(distance) > float(max_distance):
            over_distance += 1
            continue

        triangle = face_vertices[int(face_index)]
        a, b, c = triangle
        query = np.asarray(location, dtype=np.float64)
        v0 = b - a
        v1 = c - a
        v2 = query - a
        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) <= 1.0e-20:
            barycentric = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            w1 = (d11 * d20 - d01 * d21) / denominator
            w2 = (d00 * d21 - d01 * d20) / denominator
            barycentric = np.array([1.0 - w1 - w2, w1, w2], dtype=np.float64)
            barycentric = np.clip(barycentric, 0.0, 1.0)
            total = float(barycentric.sum())
            if total > 0.0:
                barycentric /= total

        face = source_faces[int(face_index)]
        out[i] = (
            barycentric[0] * source_weights[face[0]]
            + barycentric[1] * source_weights[face[1]]
            + barycentric[2] * source_weights[face[2]]
        )
        matched[i] = True
        if (i + 1) % 20000 == 0:
            print(
                f"[nearest-bvh] queried {i + 1}/{len(target_vertices)} target vertices",
                flush=True,
            )

    stats = {
        "backend": "blender_bvh",
        "target_vertices": int(len(target_vertices)),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "no_candidates": int(no_candidates),
        "over_distance": int(over_distance),
    }
    return normalize_rows(out), matched, stats


def nearest_transfer(src_mesh, tgt_mesh, armature_obj, args):
    src_vertices = mesh_vertices_world(src_mesh)
    src_faces = triangulated_faces(src_mesh)
    group_names, src_weights = source_vertex_weights(src_mesh)

    apply_transforms(tgt_mesh)
    tgt_vertices = mesh_vertices_local(tgt_mesh)
    tgt_faces = triangulated_faces(tgt_mesh)

    diag = float(np.linalg.norm(mesh_bounds(src_vertices)[1] - mesh_bounds(src_vertices)[0]))
    max_distance = None if args.max_distance_ratio <= 0 else diag * args.max_distance_ratio
    if args.nearest_backend == "bvh":
        transferred, matched, stats = transfer_weights_by_blender_bvh(
            source_vertices=src_vertices,
            source_faces=src_faces,
            source_weights=src_weights,
            target_vertices=tgt_vertices,
            max_distance=max_distance,
        )
    else:
        transferred, matched, stats = transfer_weights_by_nearest_surface(
            source_vertices=src_vertices,
            source_faces=src_faces,
            source_weights=src_weights,
            target_vertices=tgt_vertices,
            max_distance=max_distance,
            candidate_count=args.candidate_count,
        )
    print(f"[nearest-transfer] stats={stats} bbox_diag={diag:.4f} "
          f"max_distance={max_distance}", flush=True)
    filled_weights, filled_mask, n_iters = inpaint_missing_weights(
        tgt_faces,
        transferred,
        matched,
        max_iterations=96,
    )
    print(f"[nearest-inpaint] filled={int(filled_mask.sum())}/{len(filled_mask)} "
          f"iterations={n_iters}", flush=True)
    final_weights = keep_top_k_normalized(filled_weights, k=args.top_k)
    write_vertex_groups(tgt_mesh, group_names, final_weights, args.min_weight)
    parent_to_armature(tgt_mesh, armature_obj)


def main():
    args = parse_argv()
    for path in (args.rig_glb, args.new_mesh):
        if not os.path.exists(path):
            raise SystemExit(f"MISSING {path}")
    if args.new_diffuse and not os.path.exists(args.new_diffuse):
        raise SystemExit(f"MISSING {args.new_diffuse}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_rig_scene(args.rig_glb)
    original_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    original_armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not original_meshes or not original_armatures:
        raise SystemExit("rig glb must contain at least one mesh and one armature")
    src_mesh = max(original_meshes, key=lambda o: len(o.data.vertices))
    armature = original_armatures[0]
    print(f"[load] source mesh={src_mesh.name} verts={len(src_mesh.data.vertices)} "
          f"armature={armature.name} bones={len(armature.data.bones)} "
          f"actions={[a.name for a in bpy.data.actions]}", flush=True)

    new_meshes = import_mesh_only(args.new_mesh)
    if not new_meshes:
        raise SystemExit("new mesh import returned no mesh objects")
    tgt_mesh = max(new_meshes, key=lambda o: len(o.data.vertices))
    print(f"[load] target mesh={tgt_mesh.name} verts={len(tgt_mesh.data.vertices)}", flush=True)
    # FBX targets commonly inherit a unit-conversion transform from their
    # source armature. Detach while preserving world space before any weld,
    # cardinal rotation, or bbox alignment.
    if tgt_mesh.parent is not None:
        target_world = tgt_mesh.matrix_world.copy()
        tgt_mesh.parent = None
        tgt_mesh.matrix_world = target_world
        bpy.context.view_layer.update()
        print("[target-parent] detached imported parent while preserving world transform", flush=True)
    weld_target_position_duplicates(tgt_mesh)

    if args.flip_x:
        tgt_mesh.scale.x *= -1.0
        bpy.context.view_layer.update()
        apply_transforms(tgt_mesh)
        recompute_normals(tgt_mesh)
        print("[flip-x] mirrored target along X", flush=True)
    rotate_target_z_degrees(tgt_mesh, args.target_rotate_z_deg)

    if args.auto_align == "yes":
        align_bbox(tgt_mesh, src_mesh, args.align_mode)

    remove_ground_artifacts(tgt_mesh, args)
    if args.new_diffuse:
        assign_texture(tgt_mesh, args.new_diffuse, use_backface_culling=(args.backface_culling == "yes"))
    else:
        print("[texture] no --new-diffuse provided; keeping imported material", flush=True)
    if args.weight_mode == "auto":
        auto_weight_to_armature(tgt_mesh, armature)
    elif args.weight_mode == "nearest":
        nearest_transfer(src_mesh, tgt_mesh, armature, args)
    else:
        robust_transfer(src_mesh, tgt_mesh, armature, args)
        parent_to_armature(tgt_mesh, armature)
    if args.reverse_actions == "yes":
        reverse_matching_actions(args.reverse_action_hints)
    head_bones = [b.strip() for b in args.head_bones.split(",") if b.strip()]
    dampen_bone_rotation_actions(head_bones, args.dampen_head_rotations)
    tail_bones = [b.strip() for b in args.tail_bones.split(",") if b.strip()]
    dampen_bone_rotation_actions(tail_bones, args.dampen_tail_rotations)
    foot_bones = [b.strip() for b in args.foot_bones.split(",") if b.strip()]
    dampen_bone_rotation_actions(foot_bones, args.dampen_foot_rotations)
    if args.export_action_policy == "walk-idle":
        keep_canonical_walk_idle_actions(armature)
    if args.animation_export_mode == "nla":
        stash_armature_actions_for_gltf(armature)

    src_name = src_mesh.name
    bpy.data.objects.remove(src_mesh, do_unlink=True)
    print(f"[cleanup] removed original mesh {src_name}", flush=True)

    for obj in bpy.data.objects:
        obj.select_set(False)
    tgt_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS" if args.animation_export_mode == "nla" else "ACTIONS",
        export_nla_strips=args.animation_export_mode == "nla",
        export_extra_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    if args.ue_safe_animation_channels == "yes":
        keep_paths = [p.strip() for p in args.ue_animation_keep_paths.split(",") if p.strip()]
        postprocess = postprocess_glb_animation_channels(
            args.output,
            keep_paths,
            canonical_walk_idle=(args.export_action_policy == "walk-idle"),
        )
        print(
            f"[anim] UE-safe GLB channel filter keep={keep_paths} "
            f"removed={postprocess['removed_channels']} "
            f"animations={postprocess['animation_names']}",
            flush=True,
        )
    print(f"ROBUST_SWAP_OK output={args.output} verts={len(tgt_mesh.data.vertices)}", flush=True)


if __name__ == "__main__":
    main()
