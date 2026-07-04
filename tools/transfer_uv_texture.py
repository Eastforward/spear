"""Stage 2 of the animated-dog-hunyuan-paint spec.

Transfer a texture painted on Hunyuan's UV layout back to the original
mesh's UV layout via per-triangle barycentric correspondence in world
space.

Algorithm (per spec §4.3):
  1. Build KD-tree over hunyuan-mesh triangle centers (world space).
  2. For each triangle in the original mesh:
       - Find nearest hunyuan triangle by center distance.
       - Rasterize the original triangle into the output image using its
         UV coordinates as pixel positions.
       - For each output pixel, compute barycentric weights wrt the
         original triangle's world corners; apply the SAME weights to
         the hunyuan triangle's world corners (giving a world position),
         and also to the hunyuan triangle's UV corners (giving a hunyuan
         UV); sample the hunyuan diffuse at that UV to get the color.
  3. Dilate output by --dilate iterations to fill UV-atlas seam gaps.

Prints `UV_TRANSFER_OK size=NxN painted_pixels=P nonzero_fraction=F`.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-mesh", required=True, help="Original mesh (glb or obj) with UVs to paint onto")
    p.add_argument("--hy3d-mesh", required=True, help="Hunyuan white_mesh_remesh.obj with source UVs")
    p.add_argument("--hy3d-diffuse", required=True, help="Hunyuan hy3d_diffuse.jpg (source texture)")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--size", type=int, default=1024, help="Output texture size in pixels (square)")
    p.add_argument("--dilate", type=int, default=3, help="Numpy 4-neighbor dilate iterations to fill seams")
    return p.parse_args()


def load_mesh_with_uv(path):
    """Load a mesh and return (vertices, faces, uvs) where uvs is one UV per vertex.

    trimesh gives us mesh.vertices (Nx3), mesh.faces (Mx3), and
    mesh.visual.uv (Nx2). If uv is None, raise.
    """
    m = trimesh.load(path, force="mesh", process=False)
    if not hasattr(m.visual, "uv") or m.visual.uv is None:
        raise SystemExit(f"UV_TRANSFER_FAIL {path} has no UVs")
    return np.asarray(m.vertices, dtype=np.float64), \
           np.asarray(m.faces, dtype=np.int64), \
           np.asarray(m.visual.uv, dtype=np.float64)


def barycentric(px, py, x0, y0, x1, y1, x2, y2):
    """Compute barycentric coords of point (px,py) in triangle (x0..x2, y0..y2).
    Returns (w0, w1, w2). Sum ~= 1.
    """
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def dilate_once(img):
    """Grow painted region by one pixel using numpy shifts. Painted =
    pixel with any non-zero channel."""
    mask = (img.sum(axis=2) > 0)
    out = img.copy()
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
        nb = np.zeros_like(img)
        if dy == -1: nb[1:, :] = img[:-1, :]
        elif dy == 1: nb[:-1, :] = img[1:, :]
        elif dx == -1: nb[:, 1:] = img[:, :-1]
        elif dx == 1: nb[:, :-1] = img[:, 1:]
        nb_mask = (nb.sum(axis=2) > 0) & ~mask
        for c in range(3):
            out[..., c] = np.where(nb_mask, nb[..., c], out[..., c])
    return out


def main():
    args = parse_args()

    orig_v, orig_f, orig_uv = load_mesh_with_uv(args.orig_mesh)
    hy_v, hy_f, hy_uv = load_mesh_with_uv(args.hy3d_mesh)
    hy_diff = cv2.imread(args.hy3d_diffuse)
    if hy_diff is None:
        print(f"UV_TRANSFER_FAIL could not read {args.hy3d_diffuse}")
        sys.exit(1)
    hy_h, hy_w = hy_diff.shape[:2]

    # KD-tree over hunyuan triangle centers in world space
    hy_centers = (hy_v[hy_f[:, 0]] + hy_v[hy_f[:, 1]] + hy_v[hy_f[:, 2]]) / 3.0
    tree = cKDTree(hy_centers)

    size = args.size
    img = np.zeros((size, size, 3), dtype=np.float64)

    for f_idx, tri in enumerate(orig_f):
        v0, v1, v2 = orig_v[tri[0]], orig_v[tri[1]], orig_v[tri[2]]
        uv0, uv1, uv2 = orig_uv[tri[0]], orig_uv[tri[1]], orig_uv[tri[2]]
        # Find nearest hunyuan triangle
        center = (v0 + v1 + v2) / 3.0
        _, nn_idx = tree.query(center, k=1)
        htri = hy_f[nn_idx]
        hv0, hv1, hv2 = hy_v[htri[0]], hy_v[htri[1]], hy_v[htri[2]]
        huv0, huv1, huv2 = hy_uv[htri[0]], hy_uv[htri[1]], hy_uv[htri[2]]

        # Rasterize orig triangle in the output image using its UVs as pixel coords.
        # UV convention: (u=0,v=0) = bottom-left in Blender; our image (0,0) = top-left.
        def to_px(uv):
            return (uv[0] * (size - 1), (1.0 - uv[1]) * (size - 1))
        p0 = to_px(uv0); p1 = to_px(uv1); p2 = to_px(uv2)

        xs = [p0[0], p1[0], p2[0]]; ys = [p0[1], p1[1], p2[1]]
        x_min = max(0, int(math.floor(min(xs))))
        x_max = min(size - 1, int(math.ceil(max(xs))))
        y_min = max(0, int(math.floor(min(ys))))
        y_max = min(size - 1, int(math.ceil(max(ys))))
        if x_max < x_min or y_max < y_min:
            continue

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                bar = barycentric(x + 0.5, y + 0.5,
                                  p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
                if bar is None:
                    continue
                w0, w1, w2 = bar
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                # Same barycentric weights on the HUNYUAN triangle's UVs
                hu = w0 * huv0[0] + w1 * huv1[0] + w2 * huv2[0]
                hv = w0 * huv0[1] + w1 * huv1[1] + w2 * huv2[1]
                # Sample hunyuan diffuse (flip v because image origin)
                sx = int(np.clip(hu * (hy_w - 1), 0, hy_w - 1))
                sy = int(np.clip((1.0 - hv) * (hy_h - 1), 0, hy_h - 1))
                img[y, x, 0] = hy_diff[sy, sx, 0]
                img[y, x, 1] = hy_diff[sy, sx, 1]
                img[y, x, 2] = hy_diff[sy, sx, 2]

    # Dilate to close seam gaps
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(args.dilate):
        img8 = dilate_once(img8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, img8)

    painted = int((img8.sum(axis=2) > 0).sum())
    frac = painted / (size * size)
    print(f"UV_TRANSFER_OK size={size}x{size} painted_pixels={painted} nonzero_fraction={frac:.3f}")


if __name__ == "__main__":
    main()
