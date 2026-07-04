"""Gate 1 for the animated-dog-hunyuan-paint spec.

Confirms Hunyuan's internal remesh (default use_remesh=True in the paint
pipeline) did NOT drastically change the topology of our input mesh.
Probe showed 1233->1228 verts and 602->602 faces, well within tolerance.
If Hunyuan changes verts by >5% or faces at all, our UV transfer heuristic
(nearest triangle in world space) will produce garbage.

Runs in the hunyuan3d env (trimesh is installed there) or any env with
trimesh. Prints MESH_DIFF_OK <ratios> on success.
"""
import argparse
import sys

import trimesh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="Original glb (Dog_textured.glb)")
    p.add_argument("--hy3d-mesh", required=True, help="Hunyuan's white_mesh_remesh.obj")
    p.add_argument("--vert-tolerance", type=float, default=0.05,
                   help="Max allowed |1 - verts_ratio|. Default 5%%.")
    return p.parse_args()


def main():
    args = parse_args()
    orig = trimesh.load(args.original, force="mesh")
    hy = trimesh.load(args.hy3d_mesh, force="mesh")

    n_orig_v, n_orig_f = len(orig.vertices), len(orig.faces)
    n_hy_v, n_hy_f = len(hy.vertices), len(hy.faces)

    if n_orig_v == 0 or n_orig_f == 0:
        print(f"MESH_DIFF_FAIL original has zero verts/faces (v={n_orig_v}, f={n_orig_f})")
        sys.exit(1)

    verts_ratio = n_hy_v / n_orig_v
    faces_ratio = n_hy_f / n_orig_f

    if abs(1.0 - verts_ratio) > args.vert_tolerance:
        print(f"MESH_DIFF_FAIL verts_ratio={verts_ratio:.3f} outside tolerance "
              f"[{1-args.vert_tolerance:.3f}, {1+args.vert_tolerance:.3f}] "
              f"(orig_v={n_orig_v}, hy_v={n_hy_v})")
        sys.exit(1)
    if faces_ratio != 1.0:
        print(f"MESH_DIFF_FAIL faces_ratio={faces_ratio:.3f} != 1.000 "
              f"(orig_f={n_orig_f}, hy_f={n_hy_f})")
        sys.exit(1)

    print(f"MESH_DIFF_OK verts_ratio={verts_ratio:.3f} faces_ratio={faces_ratio:.3f} "
          f"(orig=({n_orig_v}v,{n_orig_f}f) hy=({n_hy_v}v,{n_hy_f}f))")


if __name__ == "__main__":
    main()
