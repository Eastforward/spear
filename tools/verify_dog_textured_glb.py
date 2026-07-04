"""Verify that Blender's Dog_textured.glb still has skin + 2 anims + UVs.

Runs in spear-env (has pygltflib). This is the out-of-process complement to
tools/blender_add_uv_and_texture.py, which cannot import pygltflib because
Blender ships its own Python.

Exit 0 + 'GLB_VERIFY_OK' on success; exit 1 + specific 'ASSERT_*' on failure.

Usage:
  spear-env/python tools/verify_dog_textured_glb.py \\
    --input  /path/Dog.glb \\
    --output /path/Dog_textured.glb
"""

import argparse
import sys

import pygltflib


def count_verts(glb_path):
    g = pygltflib.GLTF2.load(glb_path)
    prim = g.meshes[0].primitives[0]
    return g.accessors[prim.attributes.POSITION].count


def check_attrs(glb_path):
    g = pygltflib.GLTF2.load(glb_path)
    prim = g.meshes[0].primitives[0]
    return {
        "TEXCOORD_0": prim.attributes.TEXCOORD_0,
        "JOINTS_0": prim.attributes.JOINTS_0,
        "WEIGHTS_0": prim.attributes.WEIGHTS_0,
        "animations": sorted(a.name for a in (g.animations or [])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="original skinned glb (pre-Blender)")
    p.add_argument("--output", required=True, help="Blender's textured glb")
    args = p.parse_args()

    in_v = count_verts(args.input)
    out_v = count_verts(args.output)
    # Smart UV Project introduces seams, which the glTF exporter must materialise
    # by splitting shared-vertex data at seam boundaries (one glTF vertex per
    # (position, normal, UV) combination). A small INCREASE is therefore
    # expected and correct. We only fail on:
    #   - shrink (would mean geometry got dropped/merged), or
    #   - implausible blow-up (>3x, would indicate a real bug like unintended
    #     remesh).
    if out_v < in_v:
        print(f"ASSERT_VERT_COUNT_SHRUNK input={in_v} output={out_v}", flush=True)
        return 1
    if out_v > 3 * in_v:
        print(f"ASSERT_VERT_COUNT_EXPLODED input={in_v} output={out_v}", flush=True)
        return 1

    a = check_attrs(args.output)
    if a["TEXCOORD_0"] is None:
        print("ASSERT_MISSING_TEXCOORD_0", flush=True)
        return 1
    if a["JOINTS_0"] is None:
        print("ASSERT_MISSING_JOINTS_0", flush=True)
        return 1
    if a["WEIGHTS_0"] is None:
        print("ASSERT_MISSING_WEIGHTS_0", flush=True)
        return 1
    if a["animations"] != ["Idle", "Walking"]:
        print(f"ASSERT_ANIM_MISMATCH got={a['animations']}", flush=True)
        return 1

    print(f"GLB_VERIFY_OK verts={out_v} anims={a['animations']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
