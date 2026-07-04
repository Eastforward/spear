"""Blender headless: add UVs + apply a diffuse texture to a skinned glb,
without touching bones or animations. See spec 2026-07-04-animated-dog-gpurir
Component A.

Usage:
  /data/jzy/.local/bin/blender --background --python \
    /data/jzy/code/SPEAR/tools/blender_add_uv_and_texture.py -- \
    --input  /path/Dog.glb \
    --output /path/Dog_textured.glb \
    --diffuse-texture /path/dog_fur_diffuse.jpg \
    --uv-island-margin 0.02

Post-export assertions (script exits 1 on any failure):
  - Output mesh vertex count == input mesh vertex count (no unintended remesh)
  - Output glTF primitive has TEXCOORD_0
  - Output glTF primitive has JOINTS_0
  - Output glTF primitive has WEIGHTS_0
  - Output has exactly 2 animations named 'Idle' and 'Walking'
"""

import argparse
import os
import sys

import bpy  # provided by blender --background


def parse_argv():
    # Blender passes user args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--diffuse-texture", required=True)
    p.add_argument("--uv-island-margin", type=float, default=0.02)
    return p.parse_args(argv)


# NOTE: pygltflib is not installed in Blender's bundled Python. Post-export
# assertions therefore live in a separate verifier script
# (tools/verify_dog_textured_glb.py) run in spear-env after Blender finishes.
# In-Blender we only check that mesh count / material assignment / export
# succeeded at the API level; the glb-attribute assertions run out-of-process.


def main():
    args = parse_argv()
    if not os.path.exists(args.input):
        print(f"BLENDER_INPUT_MISSING {args.input}", flush=True)
        sys.exit(1)
    if not os.path.exists(args.diffuse_texture):
        print(
            f"BLENDER_TEXTURE_MISSING {args.diffuse_texture}\n"
            f"Run tools/download_polyhaven_dog_fur.py first.",
            flush=True,
        )
        sys.exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import glb
    bpy.ops.import_scene.gltf(filepath=args.input)

    # 3. Find the primary mesh — the largest one that has a material.
    # Quaternius Dog.glb ships with a second tiny 'Icosphere' (~42 verts, no
    # material) — probably an eyeball. We only re-texture the main body.
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        print("BLENDER_NO_MESHES", flush=True)
        sys.exit(1)
    mesh_objs_with_mat = [o for o in mesh_objs if o.data.materials]
    candidates = mesh_objs_with_mat if mesh_objs_with_mat else mesh_objs
    mesh = max(candidates, key=lambda o: len(o.data.vertices))
    print(
        f"PRIMARY_MESH {mesh.name} verts={len(mesh.data.vertices)}  "
        f"(other meshes left untouched: "
        f"{[o.name for o in mesh_objs if o is not mesh]})",
        flush=True,
    )

    # 4. Enter edit mode, select all faces, Smart UV Project
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        island_margin=args.uv_island_margin,
        angle_limit=1.15,  # ~66 degrees in radians
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    # 5. Build a new material with Principled BSDF + ImageTexture
    mat = bpy.data.materials.new(name="Dog_Fur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # Remove default nodes, add fresh
    for n in list(nodes):
        nodes.remove(n)
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(args.diffuse_texture)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # 6. Replace mesh's material(s) with the new one
    mesh.data.materials.clear()
    mesh.data.materials.append(mat)

    # 7. Export glb — preserve skin + animations + UVs
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        # Bundle the image inside the glb so downstream tools have a single file
        export_image_format="AUTO",
    )
    print(f"EXPORTED {args.output}", flush=True)
    # In-Blender assertion: mesh + material link
    if not mesh.data.materials or mesh.data.materials[0].name != "Dog_Fur":
        print("BLENDER_ASSERT_MATERIAL_NOT_LINKED", flush=True)
        sys.exit(1)
    print("BLENDER_OK", flush=True)


if __name__ == "__main__":
    main()
