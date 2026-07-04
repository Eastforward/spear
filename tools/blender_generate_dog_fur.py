"""Generate a procedural dog-fur diffuse texture (1K JPG) using Blender's
built-in Cycles + Voronoi/Noise nodes and bake to a flat image. No external
asset needed. CC0 by construction (our own generated pixels).

Runs headless. Output overwrites assets/textures/animal_fur/dog_fur_diffuse.jpg.

Usage:
  /data/jzy/.local/bin/blender --background --python \
    /data/jzy/code/SPEAR/tools/blender_generate_dog_fur.py -- \
    --output /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
    --size 1024 --base-color 0.42 0.28 0.16
"""

import argparse
import os
import sys

import bpy


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--base-color", type=float, nargs=3, default=[0.42, 0.28, 0.16],
                   metavar=("R", "G", "B"), help="Warm brown default")
    return p.parse_args(argv)


def main():
    args = parse_argv()
    r, g, b = args.base_color

    # Clean scene, use Cycles + Compositor to bake a procedural pattern
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.device = "CPU"

    # Enable compositor + set up nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    # Build a procedural fur-like image: base color * 2D noise (large-scale hair
    # tufts) + fine noise for strand-level roughness.
    tex_coord = tree.nodes.new("CompositorNodeImage")  # placeholder — we'll use Texture output
    # Actually, easiest reliable route: bake a texture using a Shader ->
    # Emission material on a plane, render 1 frame, save. Compositor is
    # overkill.

    # Reset — use the plane-bake approach instead
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.render.resolution_x = args.size
    bpy.context.scene.render.resolution_y = args.size
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = 95

    # Add a plane facing the camera, size 2 (fills camera)
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane = bpy.context.object

    # Camera looking straight down at plane
    bpy.ops.object.camera_add(location=(0, 0, 2), rotation=(0, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # Material: Emission shader (so no lighting needed) fed by
    # Voronoi (large hair tufts) * Noise (strand-scale variation) * base color.
    mat = bpy.data.materials.new(name="ProcFur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    # base tone
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r, g, b, 1.0)

    # Multiplicative variation: mix RGB with (voronoi * noise) to darken tips
    tex_coord = nodes.new("ShaderNodeTexCoord")

    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "F1"
    voronoi.distance = "EUCLIDEAN"
    voronoi.inputs["Scale"].default_value = 220.0  # small tufts

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.7

    mix1 = nodes.new("ShaderNodeMixRGB")
    mix1.blend_type = "MULTIPLY"
    mix1.inputs["Fac"].default_value = 0.6
    links.new(voronoi.outputs["Distance"], mix1.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix1.inputs["Color2"])

    # combine: base_color * (0.5 + 0.5 * mix1) — brightens midtones without going black
    mult = nodes.new("ShaderNodeMixRGB")
    mult.blend_type = "MULTIPLY"
    mult.inputs["Fac"].default_value = 1.0
    links.new(rgb.outputs["Color"], mult.inputs["Color1"])
    links.new(mix1.outputs["Color"], mult.inputs["Color2"])

    # brighten so it's not too dark: add a bit of the base color back
    bright = nodes.new("ShaderNodeMixRGB")
    bright.blend_type = "ADD"
    bright.inputs["Fac"].default_value = 0.5
    links.new(mult.outputs["Color"], bright.inputs["Color1"])
    links.new(rgb.outputs["Color"], bright.inputs["Color2"])

    links.new(tex_coord.outputs["UV"], voronoi.inputs["Vector"])
    links.new(tex_coord.outputs["UV"], noise.inputs["Vector"])
    links.new(bright.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Render + save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
    print(f"WROTE {args.output} ({args.size}x{args.size})", flush=True)


if __name__ == "__main__":
    main()
