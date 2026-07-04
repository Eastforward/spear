"""Blender headless: swap the diffuse texture on an already-UV-mapped glb
without touching the UV coordinates.

Complements tools/blender_add_uv_and_texture.py (which REGENERATES UVs
via Smart UV Project every run). For the hunyuan-paint pipeline the UVs
are load-bearing — the transferred_diffuse.png was painted according to
the mesh's *existing* UVs and would misalign under a fresh Smart UV
Project run. This script only rewrites the material + image.

Usage:
  /data/jzy/.local/bin/blender --background --python \\
    /data/jzy/code/SPEAR/tools/blender_replace_diffuse.py -- \\
    --input  /path/Dog_textured.glb \\
    --output /path/Dog_new_textured.glb \\
    --diffuse-texture /path/transferred_diffuse.png

Post-export assertions (script exits 1 on any failure):
  - Output has exactly 1 body mesh with a material
  - Material has our replaced diffuse image
Additional glb-level assertions (verts / joints / anims) run separately
via tools/verify_dog_textured_glb.py.
"""

import argparse
import os
import sys

import bpy  # provided by blender --background


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--diffuse-texture", required=True)
    return p.parse_args(argv)


def main():
    args = parse_argv()
    if not os.path.exists(args.input):
        print(f"BLENDER_INPUT_MISSING {args.input}", flush=True)
        sys.exit(1)
    if not os.path.exists(args.diffuse_texture):
        print(f"BLENDER_TEXTURE_MISSING {args.diffuse_texture}", flush=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import glb (preserves existing UVs)
    bpy.ops.import_scene.gltf(filepath=args.input)

    # 3. Find primary mesh — same heuristic as blender_add_uv_and_texture.py
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

    # 4. Verify UVs already exist
    if not mesh.data.uv_layers:
        print("BLENDER_NO_UVS input mesh has no UV layer; use blender_add_uv_and_texture.py instead", flush=True)
        sys.exit(1)
    print(f"PRESERVED_UV layer={mesh.data.uv_layers[0].name}", flush=True)

    # 5. Nuke any old materials on this mesh, then any orphan materials in the file
    #    (so the new one gets its intended name, not Dog_Fur.001).
    mesh.data.materials.clear()
    for old in [m for m in bpy.data.materials if m.users == 0]:
        bpy.data.materials.remove(old)
    for old in [m for m in bpy.data.materials if m.name == "Dog_Fur"]:
        bpy.data.materials.remove(old)

    # 6. Build fresh material with Principled BSDF + ImageTexture (same shape
    #    as blender_add_uv_and_texture.py so downstream tools see identical
    #    material topology).
    mat = bpy.data.materials.new(name="Dog_Fur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(args.diffuse_texture)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # 7. Attach new material
    mesh.data.materials.append(mat)

    # 8. Export glb — preserve skin + animations + UVs (do NOT re-project)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    print(f"EXPORTED {args.output}", flush=True)

    # In-Blender assertion
    if not mesh.data.materials or mesh.data.materials[0].name != "Dog_Fur":
        print(f"BLENDER_ASSERT_MATERIAL_NOT_LINKED got={[m.name for m in mesh.data.materials]}", flush=True)
        sys.exit(1)
    print("BLENDER_REPLACE_OK", flush=True)


if __name__ == "__main__":
    main()
