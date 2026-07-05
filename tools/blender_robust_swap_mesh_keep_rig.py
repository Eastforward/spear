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
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector


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
    graph_region_labels_from_capsules,
    inpaint_missing_weights,
    keep_top_k_normalized,
    mesh_bounds,
    regularize_regions_by_connected_components,
    target_region_labels_from_source_proximity,
    transfer_weights_by_region,
)


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--rig-glb", required=True,
                   help="Original animated GLB providing armature, animations, and source weights.")
    p.add_argument("--new-mesh", required=True,
                   help="Target textured mesh (.obj/.glb/.gltf).")
    p.add_argument("--new-diffuse", required=True,
                   help="Diffuse texture to assign to the target mesh.")
    p.add_argument("--output", required=True)
    p.add_argument("--auto-align", default="yes", choices=["yes", "no"])
    p.add_argument("--align-mode", default="uniform", choices=["uniform", "nonuniform"],
                   help="uniform preserves target proportions; nonuniform matches the rig bbox per axis.")
    p.add_argument("--flip-x", action="store_true",
                   help="Mirror the target along X before aligning.")
    p.add_argument("--max-distance-ratio", type=float, default=0.35,
                   help="Reject source matches farther than this fraction of the source bbox diagonal. "
                        "Use <=0 to disable distance rejection.")
    p.add_argument("--candidate-count", type=int, default=24,
                   help="Number of nearest compatible source face centers to evaluate per target vertex.")
    p.add_argument("--top-k", type=int, default=4,
                   help="Maximum bone influences to keep per target vertex.")
    p.add_argument("--min-weight", type=float, default=1e-5,
                   help="Do not write vertex-group weights below this value.")
    p.add_argument("--weight-mode", default="region", choices=["region", "auto"],
                   help="region copies compatible source weights; auto uses Blender automatic weights "
                        "against the original armature after alignment.")
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
    p.add_argument("--backface-culling", default="no", choices=["yes", "no"],
                   help="Use single-sided material export. Usually keep no for Hunyuan fur shells.")
    return p.parse_args(argv)


def import_mesh_only(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh format {ext}")
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


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


def recompute_normals(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def mesh_vertices_world(obj):
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)


def mesh_vertices_local(obj):
    return np.array([v.co for v in obj.data.vertices], dtype=np.float64)


def dog_region_coords(vertices):
    """Convert Blender dog coordinates to the helper convention: X, up, side.

    Blender is Z-up here.  The pure helper module uses X=front, Y=up, Z=side.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices[:, [0, 2, 1]]


def _semantic_region_from_point(point_world, region_bounds):
    point = dog_region_coords(np.asarray([point_world], dtype=np.float64))[0]
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


def armature_region_capsules(armature_obj, region_bounds, source_world_diag):
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
        region = _semantic_region_from_point((head + tail) * 0.5, region_bounds)
        radius = max(float(source_world_diag) * radius_by_region[region], length * 0.18)
        capsules.append(SkeletonCapsule(region=region, start=head, end=tail, radius=radius))
        counts[REGION_NAMES.get(region, str(region))] = counts.get(REGION_NAMES.get(region, str(region)), 0) + 1
    print(f"[capsules] count={len(capsules)} regions={counts}", flush=True)
    return capsules


def triangulated_faces(obj):
    faces = []
    for poly in obj.data.polygons:
        verts = list(poly.vertices)
        if len(verts) < 3:
            continue
        for i in range(1, len(verts) - 1):
            faces.append((verts[0], verts[i], verts[i + 1]))
    if not faces:
        raise SystemExit(f"mesh {obj.name} has no faces")
    return np.asarray(faces, dtype=np.int64)


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


def robust_transfer(src_mesh, tgt_mesh, armature_obj, args):
    src_vertices = mesh_vertices_world(src_mesh)
    src_faces = triangulated_faces(src_mesh)
    group_names, src_weights = source_vertex_weights(src_mesh)
    src_region_vertices = dog_region_coords(src_vertices)
    bounds = mesh_bounds(src_region_vertices)
    source_face_regions = face_region_labels(src_region_vertices, src_faces, bounds=bounds)

    apply_transforms(tgt_mesh)
    tgt_vertices = mesh_vertices_local(tgt_mesh)
    tgt_faces = triangulated_faces(tgt_mesh)
    coarse_target_regions = coarse_region_labels(dog_region_coords(tgt_vertices), bounds=bounds)

    diag = float(np.linalg.norm(mesh_bounds(src_vertices)[1] - mesh_bounds(src_vertices)[0]))
    max_distance = None if args.max_distance_ratio <= 0 else diag * args.max_distance_ratio
    if args.segmentation_mode == "skeleton-graph":
        capsules = armature_region_capsules(armature_obj, bounds, diag)
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
        if args.segmentation_mode == "proximity-components":
            target_regions, component_stats = regularize_regions_by_connected_components(
                faces=tgt_faces,
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
    print(f"[transfer] source verts={len(src_vertices)} faces={len(src_faces)} "
          f"target verts={len(tgt_vertices)} faces={len(tgt_faces)} groups={len(group_names)}", flush=True)
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
        tgt_faces,
        transferred,
        matched,
        max_iterations=96,
    )
    print(f"[inpaint] filled={int(filled_mask.sum())}/{len(filled_mask)} iterations={n_iters}", flush=True)

    final_weights = keep_top_k_normalized(filled_weights, k=args.top_k)
    summarize_region_weights(final_weights, target_regions, group_names, "target region weight mass")
    write_vertex_groups(tgt_mesh, group_names, final_weights, args.min_weight)


def main():
    args = parse_argv()
    for path in (args.rig_glb, args.new_mesh, args.new_diffuse):
        if not os.path.exists(path):
            raise SystemExit(f"MISSING {path}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.import_scene.gltf(filepath=args.rig_glb)
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

    if args.flip_x:
        tgt_mesh.scale.x *= -1.0
        bpy.context.view_layer.update()
        apply_transforms(tgt_mesh)
        recompute_normals(tgt_mesh)
        print("[flip-x] mirrored target along X", flush=True)

    if args.auto_align == "yes":
        align_bbox(tgt_mesh, src_mesh, args.align_mode)

    assign_texture(tgt_mesh, args.new_diffuse, use_backface_culling=(args.backface_culling == "yes"))
    if args.weight_mode == "auto":
        auto_weight_to_armature(tgt_mesh, armature)
    else:
        robust_transfer(src_mesh, tgt_mesh, armature, args)
        parent_to_armature(tgt_mesh, armature)
    foot_bones = [b.strip() for b in args.foot_bones.split(",") if b.strip()]
    dampen_bone_rotation_actions(foot_bones, args.dampen_foot_rotations)

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
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    print(f"ROBUST_SWAP_OK output={args.output} verts={len(tgt_mesh.data.vertices)}", flush=True)


if __name__ == "__main__":
    main()
