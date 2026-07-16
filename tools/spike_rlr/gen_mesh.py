"""Generate shoebox_v2_mesh.glb from shoebox_v2_spec.json.

Consumers:
  - B group: SPEAR/UE + RLR (via ``import habitat_sim``, audio-only).
    RLR ingests the GLB + per-triangle material index to compute per-band
    absorption / scattering / transmission.
  - C group: full-stack Habitat + RLR. Same GLB is loaded as the scene mesh
    (Habitat wraps RLR the same way B does; the RGB sensor is added on top).

The A group (SPEAR/UE + GPURIR) doesn't use this GLB — it uses the UE Level
built by build_shoebox_v2_umap.py. Both paths derive from the same SSOT
(shoebox_v2_spec.json), so geometry matches by construction.

Layout produced:
  - 6 axis-aligned walls (12 tris, 2 tris per wall).
  - 1 sofa box (12 tris).
  Total: 24 triangles.

Coordinate frame: right-handed Y-up, meters, matches spec's
``coordinate_frame``. Origin at room corner (0,0,0), +Y is "into the
window" direction (also camera-forward for view0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"
DB_PATH = REPO_ROOT / "data" / "acoustic_material_db.json"
DEFAULT_OUT_DIR = REPO_ROOT / "tmp" / "spike_rlr"


def _box_faces_inward(x0, x1, y0, y1, z0, z1):
    """Return vertices and triangles of an axis-aligned box.

    Triangles are ordered with outward-facing normals (right-hand rule).
    Order: +X face, -X face, +Y face, -Y face, +Z face, -Z face.
    12 triangles total, referenced back to 8 corner vertices.
    """
    verts = np.array(
        [
            [x0, y0, z0],  # 0
            [x1, y0, z0],  # 1
            [x1, y1, z0],  # 2
            [x0, y1, z0],  # 3
            [x0, y0, z1],  # 4
            [x1, y0, z1],  # 5
            [x1, y1, z1],  # 6
            [x0, y1, z1],  # 7
        ],
        dtype=np.float32,
    )
    # Faces stored as (face_name, [tri1, tri2]) so materials can be assigned
    # per-face.
    faces = {
        "+X": [(1, 2, 6), (1, 6, 5)],
        "-X": [(0, 4, 7), (0, 7, 3)],
        "+Y": [(3, 7, 6), (3, 6, 2)],
        "-Y": [(0, 1, 5), (0, 5, 4)],
        "+Z": [(4, 5, 6), (4, 6, 7)],
        "-Z": [(0, 3, 2), (0, 2, 1)],
    }
    return verts, faces


def _flip_normals(faces_by_dir):
    """Flip triangle winding so normals point INWARD (for room enclosure)."""
    return {d: [(a, c, b) for (a, b, c) in tris] for d, tris in faces_by_dir.items()}


def build_room_mesh(spec, db):
    """Build room walls + sofa as a single trimesh with per-face material tags.

    Returns:
        mesh: trimesh.Trimesh (24 tris for shoebox v2)
        face_material_tags: list[str] length == n_triangles, each entry is
            the material key resolvable in `db`.
    """
    rs = spec["room_size_m"]
    surfaces = spec["surfaces"]

    all_verts = []
    all_tris = []
    face_material_tags = []

    def emit_face(verts, tris_local, mat_tag):
        """Append a set of triangles into the growing mesh with correct offset."""
        offset = len(all_verts)
        all_verts.extend(verts.tolist())
        for tri in tris_local:
            all_tris.append([tri[0] + offset, tri[1] + offset, tri[2] + offset])
            face_material_tags.append(mat_tag)

    # --- ROOM ENCLOSURE ---
    # Six independent quads (not a shared cube), one per surface, so each
    # surface can carry its own material without vertex duplication tricks.
    room_verts, room_faces = _box_faces_inward(0, rs[0], 0, rs[1], 0, rs[2])
    room_faces = _flip_normals(room_faces)  # normals face inward

    # Map spec surface keys to face directions
    dir_to_surface_key = {
        "+X": "wall_east",
        "-X": "wall_west",
        "+Y": "wall_north",
        "-Y": "wall_south",
        "+Z": "ceiling",
        "-Z": "floor",
    }
    for dir_str, surface_key in dir_to_surface_key.items():
        mat_tag = surfaces[surface_key]
        emit_face(room_verts, room_faces[dir_str], mat_tag)

    # --- FURNITURE (sofa) ---
    for f in spec["furniture"]:
        if f["shape"] != "box":
            raise NotImplementedError(f"only box shape supported, got {f['shape']}")
        c = f["center_m"]
        s = f["size_m"]
        x0, x1 = c[0] - s[0] / 2, c[0] + s[0] / 2
        y0, y1 = c[1] - s[1] / 2, c[1] + s[1] / 2
        z0, z1 = c[2] - s[2] / 2, c[2] + s[2] / 2
        verts, faces_by_dir = _box_faces_inward(x0, x1, y0, y1, z0, z1)
        # Furniture normals face OUTWARD (leave default winding).
        mat_tag = f["material"]
        for dir_str, tris in faces_by_dir.items():
            emit_face(verts, tris, mat_tag)

    verts_arr = np.array(all_verts, dtype=np.float32)
    faces_arr = np.array(all_tris, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts_arr, faces=faces_arr, process=False)

    return mesh, face_material_tags


def build_rlr_materials(face_material_tags, db):
    """Produce (materials, material_indices) for RLR ingestion.

    materials is a list of dicts (unique per material tag) with the RLR
    absorption / scattering / transmission arrays. material_indices is a
    per-triangle int32 array indexing into materials.
    """
    # Collect unique tags, preserving first-seen order
    seen = {}
    order = []
    for tag in face_material_tags:
        if tag not in seen:
            seen[tag] = len(order)
            order.append(tag)

    materials = []
    for tag in order:
        mat = db[tag]
        materials.append(
            {
                "name": tag,
                "alpha": list(mat["alpha"]),
                "scattering": mat["scattering"],
                "transmission": list(mat["transmission"]),
            }
        )
    material_indices = [seen[tag] for tag in face_material_tags]

    return materials, material_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(SPEC_PATH))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    with open(args.db) as f:
        db = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, face_material_tags = build_room_mesh(spec, db)
    materials, material_indices = build_rlr_materials(face_material_tags, db)

    # Export GLB (visual-only; per-tri material index goes into JSON sidecar
    # because glTF doesn't have a standard acoustic-material extension).
    glb_path = out_dir / "shoebox_v2_mesh.glb"
    mesh.export(glb_path.as_posix())

    # Sidecar: per-triangle material index + material table (RLR-ready)
    materials_json_path = out_dir / "shoebox_v2_materials.json"
    with open(materials_json_path, "w") as f:
        json.dump(
            {
                "spec_version": spec["spec_version"],
                "n_triangles": len(face_material_tags),
                "materials": materials,
                "material_indices": material_indices,
                "face_material_tags": face_material_tags,
            },
            f,
            indent=2,
        )

    # Also OBJ+MTL for humans-in-the-loop (blender / meshlab preview)
    obj_path = out_dir / "shoebox_v2_mesh.obj"
    mesh.export(obj_path.as_posix())

    print(f"[gen_mesh] wrote {glb_path} ({len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} tris)")
    print(f"[gen_mesh] wrote {materials_json_path} "
          f"({len(materials)} materials, {len(material_indices)} tri-indices)")
    print(f"[gen_mesh] wrote {obj_path}")


if __name__ == "__main__":
    main()
