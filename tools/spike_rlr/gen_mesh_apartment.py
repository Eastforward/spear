"""Generate an RLR-consumable triangle mesh from apartment_shell_map.json.

Each shell actor's AABB becomes a 12-triangle box (inward-facing normals
for wall/floor/ceiling actors so RLR sees the interior; outward for the
smaller shell items like windows/doors/mirrors that are effectively slabs).

For simplicity and consistency with gen_mesh.py's shoebox pipeline, we
use outward normals uniformly — RLR uses double-sided ray-material
intersection so this doesn't affect correctness for closed shells.

Output:
  - <out_dir>/apartment_v1_mesh.glb    : the mesh (visual + geometry)
  - <out_dir>/apartment_v1_materials.json : per-triangle acoustic material
    indices in the same schema as tmp/spike_rlr/shoebox_v2_materials.json

Coordinate conversion: apartment_shell_map.json stores bboxes in UE cm
(with apartment's arbitrary world origin). We convert to SSOT (right-
handed Y-up meters) by:
   x_m = (x_ue_cm - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100
   y_m = -(y_ue_cm - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100    (Y flip per apartment convention)
   z_m = (z_ue_cm - APARTMENT_FLOOR_Z_UE_CM) / 100

The convention values match tools/gpurir_scenes/run_render_pass.py's
APARTMENT_MIC_ORIGIN_CM and APARTMENT_FLOOR_Z_CM constants.

Usage:
    /data/jzy/miniconda3/envs/ss2/bin/python \\
        tools/spike_rlr/gen_mesh_apartment.py \\
        --shell-json data/apartment_shell_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHELL_JSON = REPO_ROOT / "data" / "apartment_shell_map.json"
DEFAULT_DB = REPO_ROOT / "data" / "acoustic_material_db.json"
DEFAULT_OUT_DIR = REPO_ROOT / "tmp" / "spike_rlr"

APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)
APARTMENT_FLOOR_Z_UE_CM = 27.1

# shell_label -> acoustic material name (must exist in acoustic_material_db.json).
# Mapping choices: window/mirror → glass_window (only glass in db);
# door/picture → hardwood_oak (wood-like, no wood_solid available);
# curtain → carpet_thick (soft absorbent surrogate);
# wall → drywall_painted; floor → hardwood_oak; ceiling → painted_plaster;
# structural (large unnamed) → drywall_painted (conservative default).
SHELL_LABEL_TO_MATERIAL = {
    "shell_wall": "drywall_painted",
    "shell_floor": "hardwood_oak",
    "shell_ceiling": "painted_plaster",
    "shell_window": "glass_window",
    "shell_door": "hardwood_oak",
    "shell_curtain": "carpet_thick",
    "shell_picture": "hardwood_oak",
    "shell_mirror": "glass_window",
    "structural": "drywall_painted",
}


def ue_to_ssot(pos_ue_cm):
    x_cm, y_cm, z_cm = pos_ue_cm
    return (
        (x_cm - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0,
        -(y_cm - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0,
        (z_cm - APARTMENT_FLOOR_Z_UE_CM) / 100.0,
    )


def _box_triangles(lo_m, hi_m):
    """Return (vertices [8,3], faces [12,3]) for an axis-aligned box with
    outward-facing normals (CCW winding when viewed from outside).
    Matches the OBJ/glTF convention used by trimesh."""
    x0, y0, z0 = lo_m
    x1, y1, z1 = hi_m
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],   # -Z (bottom)
        [4, 5, 6], [4, 6, 7],   # +Z (top)
        [0, 1, 5], [0, 5, 4],   # -Y
        [2, 3, 7], [2, 7, 6],   # +Y
        [1, 2, 6], [1, 6, 5],   # +X
        [0, 4, 7], [0, 7, 3],   # -X
    ], dtype=np.int32)
    return v, f


def build_shell_mesh(shell_map, db):
    """Build per-actor boxes; return (mesh, face_material_tags).

    face_material_tags is a list of material-name strings, length == n_tris.
    """
    all_verts = []
    all_faces = []
    face_material_tags = []
    n_verts = 0
    n_skipped = 0

    for a in shell_map["shell_actors"]:
        bmin_ue = a["bbox_min_ue_cm"]
        bmax_ue = a["bbox_max_ue_cm"]
        bmin_s = ue_to_ssot(bmin_ue)
        bmax_s = ue_to_ssot(bmax_ue)
        lo = (min(bmin_s[0], bmax_s[0]),
              min(bmin_s[1], bmax_s[1]),
              min(bmin_s[2], bmax_s[2]))
        hi = (max(bmin_s[0], bmax_s[0]),
              max(bmin_s[1], bmax_s[1]),
              max(bmin_s[2], bmax_s[2]))
        # Skip degenerate boxes (zero volume)
        volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2])
        if volume < 1e-6:
            n_skipped += 1
            continue

        v, f = _box_triangles(lo, hi)
        all_verts.append(v)
        all_faces.append(f + n_verts)
        n_verts += 8

        label = a["shell_label"]
        material = SHELL_LABEL_TO_MATERIAL.get(label)
        if material is None:
            raise ValueError(f"no material mapping for shell_label={label!r}")
        if material not in db:
            raise ValueError(
                f"material {material!r} (for {label}) not in acoustic_material_db.json; "
                f"available: {[k for k in db if not k.startswith('_')]}"
            )
        # 12 triangles per box, all get this actor's material
        face_material_tags.extend([material] * 12)

    if not all_verts:
        raise SystemExit("[gen_mesh_apt] no valid shell actors — mesh would be empty")

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if n_skipped:
        print(f"[gen_mesh_apt] skipped {n_skipped} degenerate actor(s)")
    return mesh, face_material_tags


def build_rlr_materials(face_material_tags, db):
    """Same schema as gen_mesh.py::build_rlr_materials — RLR-ready sidecar."""
    seen = {}
    order = []
    for tag in face_material_tags:
        if tag not in seen:
            seen[tag] = len(order)
            order.append(tag)
    materials = []
    for tag in order:
        mat = db[tag]
        materials.append({
            "name": tag,
            "alpha": list(mat["alpha"]),
            "scattering": mat["scattering"],
            "transmission": list(mat["transmission"]),
        })
    material_indices = [seen[tag] for tag in face_material_tags]
    return materials, material_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-json", default=str(DEFAULT_SHELL_JSON))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--out-glb", default=None,
                    help="Explicit output .glb path (overrides --out-dir)")
    ap.add_argument("--out-materials", default=None,
                    help="Explicit output materials sidecar json path (overrides --out-dir)")
    args = ap.parse_args()

    shell_map = json.loads(Path(args.shell_json).read_text())
    db = json.loads(Path(args.db).read_text())

    mesh, face_material_tags = build_shell_mesh(shell_map, db)
    materials, material_indices = build_rlr_materials(face_material_tags, db)

    if args.out_glb is not None:
        glb_path = Path(args.out_glb)
    else:
        glb_path = Path(args.out_dir) / "apartment_v1_mesh.glb"
    if args.out_materials is not None:
        mat_path = Path(args.out_materials)
    else:
        mat_path = Path(args.out_dir) / "apartment_v1_materials.json"

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mat_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(glb_path.as_posix())
    with open(mat_path, "w") as f:
        json.dump({
            "shell_map_source": str(args.shell_json),
            "n_actors": len(shell_map["shell_actors"]),
            "n_triangles": len(face_material_tags),
            "materials": materials,
            "material_indices": material_indices,
            "face_material_tags": face_material_tags,
        }, f, indent=2)

    print(f"[gen_mesh_apt] wrote {glb_path} "
          f"({len(mesh.vertices)} verts, {len(mesh.faces)} tris)")
    print(f"[gen_mesh_apt] wrote {mat_path} "
          f"({len(materials)} materials, {len(material_indices)} tri-indices)")


if __name__ == "__main__":
    main()
