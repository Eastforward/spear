"""Blender headless: swap the mesh in an animated glb (Dog_textured.glb)
with a different, higher-poly textured mesh, while preserving:
  - armature (bones)
  - both Idle and Walking animations
  - the new mesh's UV + diffuse

Approach:
  1. Load both glbs
  2. Extract the original armature + animations
  3. Load the new textured mesh (obj + diffuse or glb)
  4. Auto-align the new mesh to the original (bbox-center + auto-scale + optional axis flip)
  5. Data-transfer vertex groups (skinning weights) from original mesh
     to new mesh (nearest-vertex mode)
  6. Parent new mesh to the original armature with vertex-group binding
  7. Export as glb

Usage:
  blender --background --python tools/blender_swap_mesh_keep_rig.py -- \\
    --rig-glb   /path/Dog_textured.glb \\
    --new-mesh  /path/collie2_textured.obj \\
    --new-diffuse /path/collie2_diffuse_corrected.png \\
    --output    /path/Dog_swap.glb \\
    [--auto-align yes]
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector, Matrix


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--rig-glb", required=True,
                   help="Original animated glb (provides armature + animations + rest-pose mesh)")
    p.add_argument("--new-mesh", required=True,
                   help="High-poly textured mesh (obj or glb)")
    p.add_argument("--new-diffuse", required=True,
                   help="Diffuse texture PNG/JPG for the new mesh")
    p.add_argument("--output", required=True)
    p.add_argument("--auto-align", default="yes", choices=["yes", "no"])
    p.add_argument("--flip-x", action="store_true",
                   help="Mirror new mesh along X (use when the new mesh's forward is opposite to the rig).")
    p.add_argument("--body-only", action="store_true",
                   help="Only swap the BODY portion (torso + legs). Head and tail are stitched from "
                        "the original mesh so their bones drive them cleanly. Uses spatial cut on X.")
    p.add_argument("--head-cut", type=float, default=0.55,
                   help="With --body-only: fraction (0..1) along X from the source mesh's back to keep "
                        "as body. Vertices past this go to head. Default 0.55.")
    p.add_argument("--tail-cut", type=float, default=0.10,
                   help="With --body-only: fraction (0..1) along X from source mesh's back below which "
                        "vertices are considered tail (dropped and refilled from original). Default 0.10.")
    return p.parse_args(argv)


def import_mesh_only(path):
    """Import a mesh file (obj or glb), return list of newly-added MESH objects."""
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh format {ext}")
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def align_bbox(target_obj, src_obj):
    """Rescale + translate target_obj so its axis-aligned bbox matches src_obj's.

    Uses UNIFORM scale (single factor across xyz) to preserve proportions,
    computed as the ratio of the LONGEST axis of both bounding boxes.
    """
    # Get world-space bounding boxes
    def bbox(o):
        pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
        mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        return mn, mx
    smn, smx = bbox(src_obj)
    tmn, tmx = bbox(target_obj)
    src_ext = smx - smn
    tgt_ext = tmx - tmn
    # Uniform scale by max extent ratio
    factor = max(src_ext) / max(tgt_ext)
    print(f"[align] src_ext={list(src_ext)} tgt_ext={list(tgt_ext)} uniform_scale={factor:.4f}",
          flush=True)
    # Apply scale in-place
    target_obj.scale = (factor, factor, factor)
    bpy.context.view_layer.update()

    # Recompute bbox after scale, then translate so centers match
    tmn, tmx = bbox(target_obj)
    src_center = (smn + smx) * 0.5
    tgt_center = (tmn + tmx) * 0.5
    delta = src_center - tgt_center
    target_obj.location += delta
    bpy.context.view_layer.update()
    print(f"[align] applied translate delta={list(delta)}", flush=True)


def apply_transforms(obj):
    """Apply object.location/rotation/scale into vertex data so nothing carries over."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def body_only_crop(tgt_obj, src_obj, head_cut, tail_cut):
    """Delete Hunyuan mesh vertices in the HEAD and TAIL X-regions, then
    merge back the equivalent regions from the ORIGINAL Dog mesh.

    The idea: skinning weight transfer is only reliable in regions where
    the two meshes have similar shape. Head + tail differ most between
    the low-poly Quaternius Dog and Hunyuan's realistic sculpt, so the
    weights would map incorrectly there. By using the original for head
    and tail (which already has correct weights), and Hunyuan just for
    the body (where both meshes are similar cylinders), we get high-poly
    body detail without head/tail distortion.

    Returns the target mesh object (may be the same reference — it's
    edited in-place).
    """
    import numpy as np
    # 1. Compute src bbox on X to derive absolute cut planes
    def bbox_x(obj):
        pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        return min(p.x for p in pts), max(p.x for p in pts)
    src_x_min, src_x_max = bbox_x(src_obj)
    x_range = src_x_max - src_x_min
    tail_x = src_x_min + tail_cut * x_range
    head_x = src_x_min + head_cut * x_range
    print(f"[body-only] src X range=[{src_x_min:.2f}, {src_x_max:.2f}]  "
          f"tail_cut_x={tail_x:.2f}  head_cut_x={head_x:.2f}", flush=True)

    # 2. Delete target mesh vertices outside [tail_x, head_x] X range
    bpy.context.view_layer.objects.active = tgt_obj
    tgt_obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    verts_to_remove = []
    for i, v in enumerate(tgt_obj.data.vertices):
        wp = tgt_obj.matrix_world @ v.co
        if wp.x < tail_x or wp.x > head_x:
            verts_to_remove.append(i)
    print(f"[body-only] removing {len(verts_to_remove)}/{len(tgt_obj.data.vertices)} target verts "
          f"outside body X range", flush=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for i in verts_to_remove:
        tgt_obj.data.vertices[i].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 3. Duplicate src object, delete its BODY (keep head + tail)
    bpy.ops.object.select_all(action="DESELECT")
    src_obj.select_set(True)
    bpy.context.view_layer.objects.active = src_obj
    bpy.ops.object.duplicate(linked=False)
    src_dup = bpy.context.active_object
    src_dup.name = f"{src_obj.name}_headtail"

    # Delete verts INSIDE body range on the duplicate
    verts_body = []
    for i, v in enumerate(src_dup.data.vertices):
        wp = src_dup.matrix_world @ v.co
        if tail_x <= wp.x <= head_x:
            verts_body.append(i)
    print(f"[body-only] on src copy, removing {len(verts_body)}/{len(src_dup.data.vertices)} verts "
          f"inside body X range (keeping head+tail)", flush=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for i in verts_body:
        src_dup.data.vertices[i].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 4. Join src_dup INTO tgt_obj so we end up with a single hybrid mesh
    #    (tgt keeps its material; src_dup's material comes along as slot 2).
    bpy.ops.object.select_all(action="DESELECT")
    src_dup.select_set(True)
    tgt_obj.select_set(True)
    bpy.context.view_layer.objects.active = tgt_obj
    bpy.ops.object.join()
    print(f"[body-only] hybrid mesh: verts={len(tgt_obj.data.vertices)} "
          f"faces={len(tgt_obj.data.polygons)}", flush=True)
    return tgt_obj


def transfer_vertex_groups(src_obj, tgt_obj):
    """Copy src_obj's vertex groups (bone weights) onto tgt_obj via
    Blender's Data Transfer modifier (nearest vertex mode).

    Both objects should be in the same coordinate space (aligned) BEFORE
    calling this.
    """
    # Ensure both have their transforms applied
    apply_transforms(src_obj)
    apply_transforms(tgt_obj)

    # Create empty vertex groups on target matching source (so DataTransfer
    # can populate them by name)
    for vg in src_obj.vertex_groups:
        if vg.name not in tgt_obj.vertex_groups:
            tgt_obj.vertex_groups.new(name=vg.name)

    # Add Data Transfer modifier on target
    bpy.context.view_layer.objects.active = tgt_obj
    mod = tgt_obj.modifiers.new(name="WeightTransfer", type="DATA_TRANSFER")
    mod.object = src_obj
    # POLYINTERP_NEAREST: for each target vertex, find nearest triangle on
    # source and barycentric-interpolate weights within it. Smooth, no
    # per-vertex discontinuities. Prior attempt used NEAREST (vertex→vertex)
    # which caused whole clusters of target verts to snap to a single source
    # vertex's bone and animate as a rigid block. POLYINTERP_NEAREST fixes
    # that.
    mod.use_loop_data = False
    mod.use_poly_data = False
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"

    # Apply modifier
    with bpy.context.temp_override(object=tgt_obj):
        bpy.ops.object.modifier_apply(modifier="WeightTransfer")
    print(f"[transfer] transferred {len(src_obj.vertex_groups)} vertex groups "
          f"via POLYINTERP_NEAREST mapping", flush=True)


def parent_to_armature(mesh_obj, armature_obj):
    """Parent mesh to armature with ARMATURE_NAME modifier."""
    mesh_obj.parent = armature_obj
    mesh_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = armature_obj
    mod.use_vertex_groups = True
    print(f"[parent] mesh {mesh_obj.name} -> armature {armature_obj.name} (vertex groups)",
          flush=True)


def assign_texture(mesh_obj, diffuse_path):
    """Replace mesh's material with a Principled BSDF + ImageTexture material
    using the given diffuse image."""
    for old in [m for m in bpy.data.materials if m.users == 0]:
        bpy.data.materials.remove(old)
    for old in [m for m in bpy.data.materials if m.name in ("Dog_Fur", "Swapped_Fur")]:
        bpy.data.materials.remove(old)
    mat = bpy.data.materials.new(name="Swapped_Fur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(diffuse_path)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    print(f"[texture] set diffuse {diffuse_path} on {mesh_obj.name}", flush=True)


def main():
    args = parse_argv()
    for p in (args.rig_glb, args.new_mesh, args.new_diffuse):
        if not os.path.exists(p):
            print(f"MISSING {p}", flush=True); sys.exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Load original glb (armature + anims + rest-mesh)
    bpy.ops.import_scene.gltf(filepath=args.rig_glb)
    original_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    original_armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not original_meshes or not original_armatures:
        print("RIG_GLB_MISSING mesh or armature")
        print(f"  meshes={[o.name for o in original_meshes]}  armatures={[o.name for o in original_armatures]}")
        sys.exit(1)
    src_mesh = max(original_meshes, key=lambda o: len(o.data.vertices))
    armature = original_armatures[0]
    print(f"[load] rig_glb: src_mesh={src_mesh.name} verts={len(src_mesh.data.vertices)}  "
          f"armature={armature.name}  n_bones={len(armature.data.bones)}", flush=True)
    n_anims = len(bpy.data.actions)
    print(f"[load] n_actions={n_anims} names={[a.name for a in bpy.data.actions]}", flush=True)

    # 3. Load new mesh (as sibling object)
    new_meshes = import_mesh_only(args.new_mesh)
    if not new_meshes:
        print("NEW_MESH import returned no MESH objects"); sys.exit(1)
    tgt_mesh = max(new_meshes, key=lambda o: len(o.data.vertices))
    print(f"[load] new_mesh={tgt_mesh.name} verts={len(tgt_mesh.data.vertices)}", flush=True)

    # 3.5. Optionally flip X to match src orientation (rig assumes +X = forward,
    # some meshes come with -X = forward).
    if args.flip_x:
        tgt_mesh.scale.x *= -1
        bpy.context.view_layer.update()
        # Apply the negative scale so vertices flip in-place (avoids inside-out normals)
        apply_transforms(tgt_mesh)
        # Recompute normals (they got flipped by the mirror)
        bpy.context.view_layer.objects.active = tgt_mesh
        tgt_mesh.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        print(f"[flip-x] mirrored target mesh along X and recomputed normals", flush=True)

    # 4. Auto-align target mesh into source mesh's bbox
    if args.auto_align == "yes":
        align_bbox(tgt_mesh, src_mesh)

    # 4.5. Body-only mode: keep only mid-X body portion of new mesh, then
    # append original mesh's head + tail vertices as a separate island.
    if args.body_only:
        tgt_mesh = body_only_crop(tgt_mesh, src_mesh, args.head_cut, args.tail_cut)

    # 5. Assign diffuse texture
    assign_texture(tgt_mesh, args.new_diffuse)

    # 6. Transfer vertex groups (bone weights) from src to tgt
    transfer_vertex_groups(src_mesh, tgt_mesh)

    # 7. Parent new mesh to armature
    parent_to_armature(tgt_mesh, armature)

    # 8. Remove the source (original) mesh — keep armature + tgt_mesh
    src_name = src_mesh.name
    bpy.data.objects.remove(src_mesh, do_unlink=True)
    print(f"[cleanup] removed original mesh {src_name}", flush=True)

    # 9. Export
    # Select only mesh + armature so we don't accidentally include orphans
    for o in bpy.data.objects:
        o.select_set(False)
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
    print(f"EXPORTED {args.output}", flush=True)
    print(f"SWAP_OK new_verts={len(tgt_mesh.data.vertices)} armature={armature.name} "
          f"n_anims={n_anims}", flush=True)


if __name__ == "__main__":
    main()
