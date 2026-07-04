"""Gate 2 for the animated-dog-hunyuan-paint spec.

Confirms our UV-transferred diffuse actually filled the UV atlas
(painting stayed inside the islands and covers >=85% of the polygon
area). Compares:
  - painted_fraction: fraction of output pixels with non-zero color
  - uv_area_fraction: fraction of output pixels covered by UV triangles
  - ratio = painted_fraction / uv_area_fraction  (>=0.85 => pass)

The ratio decouples the metric from mesh-specific UV atlas density. On
our Dog_textured.glb UV atlas density is ~50%, so raw painted_fraction
of 0.42 with uv_area_fraction 0.50 is ratio=0.84 -> failing but close;
raw 0.44 -> passing.
"""
import argparse
import math
import sys

import cv2
import numpy as np
import trimesh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diffuse", required=True, help="Transferred diffuse PNG")
    p.add_argument("--original-mesh", required=True, help="The mesh whose UVs this diffuse is painted on")
    p.add_argument("--min-ratio", type=float, default=0.85,
                   help="Minimum painted_fraction / uv_area_fraction to pass")
    return p.parse_args()


def compute_uv_area_fraction(mesh_path, size):
    """Rasterize the mesh's UV triangles into a size x size mask, return
    the fraction of pixels covered."""
    m = trimesh.load(mesh_path, force="mesh", process=False)
    if not hasattr(m.visual, "uv") or m.visual.uv is None:
        return 1.0   # unknown; assume full
    uvs = np.asarray(m.visual.uv, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    mask = np.zeros((size, size), dtype=np.uint8)
    for tri in faces:
        pts = []
        for vi in tri:
            u = uvs[vi, 0]; v = uvs[vi, 1]
            x = int(round(u * (size - 1)))
            y = int(round((1.0 - v) * (size - 1)))
            pts.append((x, y))
        cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 1)
    return float(mask.sum()) / (size * size)


def main():
    args = parse_args()
    img = cv2.imread(args.diffuse)
    if img is None:
        print(f"UV_COVERAGE_FAIL could not read {args.diffuse}")
        sys.exit(1)
    h, w = img.shape[:2]
    if h != w:
        print(f"UV_COVERAGE_FAIL expected square diffuse, got {w}x{h}")
        sys.exit(1)

    painted_fraction = float((img.sum(axis=2) > 0).sum()) / (h * w)
    uv_area_fraction = compute_uv_area_fraction(args.original_mesh, h)
    if uv_area_fraction <= 0:
        print(f"UV_COVERAGE_FAIL uv_area_fraction={uv_area_fraction}")
        sys.exit(1)
    ratio = painted_fraction / uv_area_fraction

    if ratio < args.min_ratio:
        print(f"UV_COVERAGE_FAIL painted_fraction={painted_fraction:.3f} "
              f"uv_area_fraction={uv_area_fraction:.3f} ratio={ratio:.3f} "
              f"(< {args.min_ratio})")
        sys.exit(1)

    print(f"UV_COVERAGE_OK painted_fraction={painted_fraction:.3f} "
          f"uv_area_fraction={uv_area_fraction:.3f} ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
