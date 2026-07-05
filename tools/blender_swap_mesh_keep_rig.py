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

    # 4. Auto-align target mesh into source mesh's bbox
    if args.auto_align == "yes":
        align_bbox(tgt_mesh, src_mesh)

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
