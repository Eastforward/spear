"""Escape hatch (spec §6 F4) for the animated-dog-hunyuan-paint spec.

Applies Blender's Catmull-Clark Subdivision Surface modifier to
Dog_textured.glb. Skinning weights propagate automatically to new
vertices via Blender's default subdivision behavior; animation data is
preserved because glb-side AnimSequences reference bone names, not
vertex indices.

Runs in Blender's bundled Python:
  blender --background --python tools/blender_subdivide_dog.py -- \\
    --input INPUT.glb --output OUTPUT.glb --levels 1
"""
import argparse
import os
import sys

import bpy


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--levels", type=int, default=1, choices=[1, 2])
    return p.parse_args(argv)


def main():
    args = parse_argv()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)

    # Pick the largest mesh (dog body), leave others (e.g. eyeball) untouched
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    if not meshes:
        print("SUBDIVIDE_FAIL no MESH objects in input")
        sys.exit(1)
    body = meshes[0]
    print(f"[subdivide] body mesh: {body.name} verts={len(body.data.vertices)} "
          f"faces={len(body.data.polygons)}", flush=True)

    # Add Subdivision Surface modifier
    mod = body.modifiers.new(name="Subdivision", type="SUBSURF")
    mod.levels = args.levels
    mod.render_levels = args.levels
    mod.subdivision_type = "CATMULL_CLARK"

    # Ensure the modifier is at the top of the stack (before Armature)
    while body.modifiers.find("Subdivision") > 0:
        with bpy.context.temp_override(object=body):
            bpy.ops.object.modifier_move_up(modifier="Subdivision")

    # Apply it
    bpy.context.view_layer.objects.active = body
    with bpy.context.temp_override(object=body):
        bpy.ops.object.modifier_apply(modifier="Subdivision")

    print(f"[subdivide] after apply: verts={len(body.data.vertices)} "
          f"faces={len(body.data.polygons)}", flush=True)

    # Export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    print(f"SUBDIVIDE_OK verts={len(body.data.vertices)} faces={len(body.data.polygons)}")


if __name__ == "__main__":
    main()
