"""Run ONCE to produce the fixture meshes + reference diffuse for tests.
Meshes: two unit squares (each 2 triangles), both spanning world x=[0,1]
y=[0,1] z=0, but with DIFFERENT UV layouts:
  mesh_a (mimics 'original'): UV = xy directly (square fills [0,1]^2)
  mesh_b (mimics 'hunyuan'): UV = xy scaled+shifted (square fills [0.25, 0.75]^2)

Reference diffuse: a 32x32 image where mesh_b's UV region [0.25,0.75]^2
(pixels [8..24, 8..24]) is red on left half, green on right half.

Expected transfer: on mesh_a's UV [0,1]^2 the output should be red on
the left half, green on the right half, painted across the FULL image
(since mesh_a's UVs span [0,1] fully).
"""
import numpy as np
import trimesh
import cv2
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Both meshes: same geometry — a unit square as 2 triangles
verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
faces = np.array([[0,1,2],[0,2,3]], dtype=np.int64)

# mesh_a UVs — full [0,1]^2
uvs_a = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
# mesh_b UVs — [0.25, 0.75]^2
uvs_b = 0.25 + 0.5 * uvs_a

for label, uvs in [("mesh_a.obj", uvs_a), ("mesh_b.obj", uvs_b)]:
    lines = []
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for uv in uvs:
        lines.append(f"vt {uv[0]} {uv[1]}")
    for f in faces:
        lines.append(f"f " + " ".join(f"{i+1}/{i+1}" for i in f))
    with open(os.path.join(HERE, label), "w") as fp:
        fp.write("\n".join(lines) + "\n")

# Reference diffuse: 32x32, red left of x=16 in the [8..24, 8..24] window, green right
img = np.zeros((32, 32, 3), dtype=np.uint8)
img[8:24, 8:16] = [0, 0, 255]      # BGR red
img[8:24, 16:24] = [0, 255, 0]     # BGR green
# Outside the [8..24, 8..24] window stays black — that's the "not painted by hunyuan" zone
cv2.imwrite(os.path.join(HERE, "hunyuan_diffuse.png"), img)

print("FIXTURE_BUILD_OK")
