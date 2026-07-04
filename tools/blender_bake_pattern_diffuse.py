"""Automated Tier-2 breed diffuse baker.

Takes the already-UV-mapped Dog_textured.glb and bakes a NEW diffuse
texture where pixel color depends on the world-space Y coordinate of
each vertex (dog's back vs belly). This yields patterns like:

  Border Collie: back = black, belly + legs + tail-tip + chest = white
  Bernese:       back = black, cheeks + legs + chest = tan/white  (3-tone)
  Dalmatian:     white base + procedural black spots
  Panda dog:     alternating black/white bands

We do this WITHOUT re-topologizing (no vertex adds/removes/reweighting),
so the animation and skinning are guaranteed intact.

Approach:
1. Load Dog_textured.glb (already has Smart-UV UVs from prior step).
2. For each polygon in the primary skinned mesh, sample its 3D center Y
   and its UV center, and paint the resulting pixel(s) in a fresh 1024x1024
   diffuse image directly.  (We bypass Blender's bake API entirely — it's
   simpler and 100% deterministic to walk polygons and paint.)
3. Anti-alias the boundary by supersampling per-pixel: for each pixel in
   UV space, look up the world Y of its corresponding surface point using
   Blender's raycast/BVH.
4. Save as PNG, then export the glb pointing at the new diffuse.

Blender axis convention here: +Z is UP (dog stands on Z=0, back is at
high Z, belly at low Z). We use local vertex Z relative to the mesh
bounding box.

Usage:
  blender --background --python tools/blender_bake_pattern_diffuse.py -- \\
    --input  tmp/animated_dog/Dog_textured.glb \\
    --pattern border_collie \\
    --output-diffuse assets/textures/dogs/border_collie_diffuse.png \\
    --output-glb tmp/animated_dog/Dog_border_collie.glb \\
    --size 1024
"""
import argparse
import math
import os
import sys

import bpy
import mathutils


PATTERNS = {
    # border_collie: back = black, belly + legs = white, cheeks white
    "border_collie": {
        "top_color":     (0.02, 0.02, 0.02),   # black
        "bottom_color":  (0.95, 0.95, 0.92),   # off-white
        "band_low":      0.35,   # below this fraction of z-range = pure bottom
        "band_high":     0.60,   # above this = pure top
    },
    # bernese: 3-tone. back black, middle band rust/tan, belly white
    "bernese": {
        "top_color":     (0.03, 0.03, 0.03),
        "middle_color":  (0.45, 0.15, 0.05),   # rust
        "bottom_color":  (0.90, 0.88, 0.80),
        "band_low":      0.30,
        "band_high":     0.65,
    },
    # panda: 3 bands, wide. back black, middle white, belly black (mostly aesthetic)
    "panda": {
        "top_color":     (0.02, 0.02, 0.02),
        "middle_color":  (0.95, 0.95, 0.92),
        "bottom_color":  (0.02, 0.02, 0.02),
        "band_low":      0.30,
        "band_high":     0.65,
    },
}


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input Dog_textured.glb (must already have UVs)")
    p.add_argument("--pattern", required=True, choices=sorted(PATTERNS.keys()))
    p.add_argument("--output-diffuse", required=True, help="Output diffuse PNG path")
    p.add_argument("--output-glb", required=True, help="Output glb pointing at the new diffuse")
    p.add_argument("--size", type=int, default=1024)
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_primary_mesh():
    """Return the largest MESH object that has a material slot (the dog body)."""
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.materials]
    if not meshes:
        raise RuntimeError("no material-bearing MESH object found")
    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    return meshes[0]


def bake_diffuse_from_polygons(mesh_obj, size, pattern_spec, output_path):
    """Walk every triangle, work out its Y (up-axis) fraction in local bbox,
    look up the target color from the pattern, and paint every UV pixel
    that this triangle covers.

    This is a pure Python software rasterizer — no Cycles bake pass, no
    scene setup. Deterministic, and doesn't need the Blender bake system
    (which has been flaky in our headless setup).
    """
    import numpy as np

    mesh = mesh_obj.data
    if not mesh.uv_layers:
        raise RuntimeError("mesh has no UV layer — did you skip blender_add_uv_and_texture.py?")
    uv_layer = mesh.uv_layers.active.data

    # In Blender +Z is up. Also apply the object's world transform in case
    # the glb import parented it under a rotation.
    world = mesh_obj.matrix_world
    verts_world = np.array([world @ v.co for v in mesh.vertices], dtype=np.float32)
    z_vals = verts_world[:, 2]
    z_min, z_max = float(z_vals.min()), float(z_vals.max())
    z_range = z_max - z_min if z_max > z_min else 1e-6

    # Precompute per-polygon color
    def color_for_zfrac(zf):
        band_low = pattern_spec["band_low"]
        band_high = pattern_spec["band_high"]
        top = np.array(pattern_spec["top_color"])
        bottom = np.array(pattern_spec["bottom_color"])
        middle = np.array(pattern_spec.get("middle_color", top))
        has_mid = "middle_color" in pattern_spec
        if not has_mid:
            # simple 2-band with smooth boundary
            if zf <= band_low:
                return bottom
            if zf >= band_high:
                return top
            t = (zf - band_low) / (band_high - band_low)
            return bottom * (1 - t) + top * t
        # 3-band: bottom | middle | top
        mid_center = 0.5 * (band_low + band_high)
        if zf <= band_low:
            return bottom
        if zf >= band_high:
            return top
        if zf <= mid_center:
            t = (zf - band_low) / (mid_center - band_low)
            return bottom * (1 - t) + middle * t
        t = (zf - mid_center) / (band_high - mid_center)
        return middle * (1 - t) + top * t

    # Compute per-face color from mean vertex Z of that face
    face_colors = []
    for poly in mesh.polygons:
        zs = [verts_world[vi, 2] for vi in poly.vertices]
        z_mean = float(np.mean(zs))
        zf = (z_mean - z_min) / z_range
        face_colors.append(color_for_zfrac(zf))

    # Rasterize each triangle into the UV image using barycentric fill.
    img = np.zeros((size, size, 4), dtype=np.float32)
    img[..., 3] = 1.0   # alpha

    def edge(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    total_polys = len(mesh.polygons)
    print(f"[bake_pattern] rasterizing {total_polys} polys into {size}x{size} diffuse", flush=True)
    for poly_idx, poly in enumerate(mesh.polygons):
        if poly.loop_total < 3:
            continue
        col = face_colors[poly_idx]
        # Grab UVs for the poly's loops
        loops = list(range(poly.loop_start, poly.loop_start + poly.loop_total))
        uvs = [uv_layer[li].uv for li in loops]
        # Fan-triangulate (glb triangulated, so this is 1 triangle usually)
        for k in range(1, poly.loop_total - 1):
            uv0 = uvs[0]
            uv1 = uvs[k]
            uv2 = uvs[k + 1]
            # convert to pixel coords; UV origin (0,0) = bottom-left in
            # Blender; our numpy image has (0,0) = top-left. Flip Y.
            def to_px(u):
                return (u[0] * (size - 1), (1.0 - u[1]) * (size - 1))
            p0 = to_px(uv0); p1 = to_px(uv1); p2 = to_px(uv2)
            # Bounding box, clipped to image
            xs = [p0[0], p1[0], p2[0]]
            ys = [p0[1], p1[1], p2[1]]
            x_min = max(0, int(math.floor(min(xs))))
            x_max = min(size - 1, int(math.ceil(max(xs))))
            y_min = max(0, int(math.floor(min(ys))))
            y_max = min(size - 1, int(math.ceil(max(ys))))
            if x_max < x_min or y_max < y_min:
                continue
            area = edge(p0, p1, p2)
            if abs(area) < 1e-9:
                continue
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    p = (x + 0.5, y + 0.5)
                    w0 = edge(p1, p2, p)
                    w1 = edge(p2, p0, p)
                    w2 = edge(p0, p1, p)
                    # inside triangle if all same sign as area
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0 and area > 0) or \
                       (w0 <= 0 and w1 <= 0 and w2 <= 0 and area < 0):
                        img[y, x, 0] = col[0]
                        img[y, x, 1] = col[1]
                        img[y, x, 2] = col[2]

    # Grow the painted region by a few pixels to cover UV bleed (avoids
    # dark pixel-edge artifacts where UV islands don't quite reach the
    # texel boundary). Pure-numpy morph dilate via 3x3 max over
    # {up,down,left,right,center} shifts. Blender ships no scipy.
    def dilate_once(a):
        # a: HxWx3 float, painted pixels have >0 sum, unpainted are exact zero
        m = (a.sum(axis=2) > 0)
        # Shift neighbors and OR into unpainted
        out = a.copy()
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = np.zeros_like(a)
            if dy == -1: nb[1:, :] = a[:-1, :]
            elif dy == 1: nb[:-1, :] = a[1:, :]
            elif dx == -1: nb[:, 1:] = a[:, :-1]
            elif dx == 1: nb[:, :-1] = a[:, 1:]
            nb_mask = (nb.sum(axis=2) > 0) & ~m
            for c in range(3):
                out[..., c] = np.where(nb_mask, nb[..., c], out[..., c])
        return out
    for _ in range(3):
        img[..., :3] = dilate_once(img[..., :3])

    # Save as 8-bit PNG using Blender's own Image API (no external deps).
    bpy_img = bpy.data.images.new(
        name="pattern_diffuse", width=size, height=size, alpha=True,
    )
    # Blender stores pixels bottom-to-top in a flat float RGBA list.
    img_flip = img[::-1]  # flip vertically so bottom-left is Blender's (0,0)
    bpy_img.pixels = img_flip.reshape(-1).tolist()
    bpy_img.filepath_raw = output_path
    bpy_img.file_format = "PNG"
    bpy_img.save()
    print(f"[bake_pattern] wrote {output_path} ({size}x{size})", flush=True)


def replace_diffuse_and_export(mesh_obj, new_diffuse_path, output_glb_path):
    """Point the primary mesh's material at new_diffuse_path (replacing
    whatever ImageTexture node currently feeds Principled/Emission), and
    export the whole scene to glb with animations+skins."""
    # Ensure a material exists and is a node tree
    mat = mesh_obj.data.materials[0]
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Load the new image
    img = bpy.data.images.load(new_diffuse_path, check_existing=False)
    img.colorspace_settings.name = "sRGB"

    # Find or create ImageTexture node and set its image
    tex_node = None
    for n in nodes:
        if n.type == "TEX_IMAGE":
            tex_node = n
            break
    if tex_node is None:
        tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img

    # Find principled BSDF and link tex.Color -> Base Color
    bsdf = None
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n; break
    if bsdf is None:
        # inject one
        for n in list(nodes):
            if n.type == "OUTPUT_MATERIAL":
                out = n; break
        else:
            out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    # Kill any existing Base Color links, then link ours
    for L in list(links):
        if L.to_node == bsdf and L.to_socket.name == "Base Color":
            links.remove(L)
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    print(f"[bake_pattern] material now points at {new_diffuse_path}", flush=True)

    bpy.ops.export_scene.gltf(
        filepath=output_glb_path,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    print(f"[bake_pattern] wrote {output_glb_path}", flush=True)


def main():
    args = parse_argv()
    if args.pattern not in PATTERNS:
        raise SystemExit(f"unknown pattern {args.pattern!r}; choose from {list(PATTERNS)}")
    pattern_spec = PATTERNS[args.pattern]

    os.makedirs(os.path.dirname(args.output_diffuse) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_glb) or ".", exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)
    mesh_obj = find_primary_mesh()
    print(f"[bake_pattern] primary mesh: {mesh_obj.name}  verts={len(mesh_obj.data.vertices)}  "
          f"polys={len(mesh_obj.data.polygons)}", flush=True)

    bake_diffuse_from_polygons(mesh_obj, args.size, pattern_spec, args.output_diffuse)
    replace_diffuse_and_export(mesh_obj, args.output_diffuse, args.output_glb)

    print("BAKE_PATTERN_OK", flush=True)


if __name__ == "__main__":
    main()
