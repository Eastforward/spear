"""Blender background script: derive a lighter runtime mesh from an approved mesh.

Run via:
  blender --background --python tools/blender_create_runtime_proxy_mesh.py -- \
      --source approved/tag/mesh_oriented.glb \
      --output approved/tag/mesh_runtime.glb \
      --metadata approved/tag/mesh_runtime.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bmesh
import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
SPIKE_RLR_DIR = SCRIPT_DIR / "spike_rlr"
if str(SPIKE_RLR_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_RLR_DIR))

from runtime_proxy_mesh import DEFAULT_TARGET_FACES, write_runtime_proxy_record  # noqa: E402


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    p.add_argument(
        "--double-sided",
        action="store_true",
        help="Export imported materials as double-sided. Useful for image-to-3D "
        "meshes whose decimated local shells contain inconsistent winding.",
    )
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _face_count(objects):
    return sum(len(obj.data.polygons) for obj in objects)


def _vertex_count(objects):
    return sum(len(obj.data.vertices) for obj in objects)


def _delete_loose_vertices(objects):
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_loose()
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")


def _topology_stats(objects):
    stats = {
        "vertices": 0,
        "edges": 0,
        "faces": 0,
        "boundary_edges": 0,
        "wire_edges": 0,
        "nonmanifold_edges_over_two_faces": 0,
        "noncontiguous_two_face_edges": 0,
    }
    for obj in objects:
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            stats["vertices"] += len(bm.verts)
            stats["edges"] += len(bm.edges)
            stats["faces"] += len(bm.faces)
            for edge in bm.edges:
                linked_faces = len(edge.link_faces)
                if linked_faces == 0:
                    stats["wire_edges"] += 1
                elif linked_faces == 1:
                    stats["boundary_edges"] += 1
                elif linked_faces > 2:
                    stats["nonmanifold_edges_over_two_faces"] += 1
                elif not edge.is_contiguous:
                    stats["noncontiguous_two_face_edges"] += 1
        finally:
            bm.free()
    return stats


def _weld_position_duplicates(objects):
    """Join glTF UV/normal split vertices before topology-aware decimation.

    glTF stores a separate vertex for each differing UV/normal tuple.  Blender's
    collapse modifier otherwise treats those coincident vertices as unrelated
    local shells and opens tens of thousands of cracks while reducing Pixal
    meshes.  UV coordinates remain per face corner in Blender, so welding the
    geometry does not collapse the texture layout.
    """
    records = []
    for obj in objects:
        before = len(obj.data.vertices)
        if before == 0:
            continue
        coordinates = [vertex.co for vertex in obj.data.vertices]
        extent = max(
            max(point[axis] for point in coordinates)
            - min(point[axis] for point in coordinates)
            for axis in range(3)
        )
        distance = max(float(extent) * 1.0e-7, 1.0e-9)
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
            bm.to_mesh(obj.data)
        finally:
            bm.free()
        obj.data.update()
        after = len(obj.data.vertices)
        records.append(
            {
                "object": obj.name,
                "vertices_before": before,
                "vertices_after": after,
                "vertices_welded": before - after,
                "distance": distance,
            }
        )
        print(
            f"[runtime_proxy] position weld object={obj.name} "
            f"vertices={before}->{after} distance={distance:.9g}",
            flush=True,
        )
    return {
        "objects": records,
        "vertices_before": sum(record["vertices_before"] for record in records),
        "vertices_after": sum(record["vertices_after"] for record in records),
        "vertices_welded": sum(record["vertices_welded"] for record in records),
    }


def _apply_decimate(objects, ratio):
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new(name="RuntimeProxyDecimate", type="DECIMATE")
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)


def _make_materials_double_sided(objects):
    changed = set()
    for obj in objects:
        for material in obj.data.materials:
            if material is None or material.name in changed:
                continue
            material.use_backface_culling = False
            changed.add(material.name)
    print(f"[runtime_proxy] double-sided materials={sorted(changed)}", flush=True)


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    metadata = Path(args.metadata)
    target_faces = int(args.target_faces)

    if target_faces <= 0:
        raise SystemExit("--target-faces must be positive")
    if not source.exists():
        raise SystemExit(f"source mesh missing: {source}")

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = _mesh_objects()
    if not meshes:
        raise SystemExit(f"no mesh objects imported from {source}")

    source_faces = _face_count(meshes)
    source_vertices = _vertex_count(meshes)
    source_topology = _topology_stats(meshes)
    weld = _weld_position_duplicates(meshes)
    welded_topology = _topology_stats(meshes)
    ratio = min(1.0, float(target_faces) / max(float(source_faces), 1.0))
    if ratio < 0.999:
        print(
            f"[runtime_proxy] decimating faces {source_faces} -> target {target_faces} "
            f"(ratio={ratio:.4f})",
            flush=True,
        )
        _apply_decimate(meshes, ratio)
    else:
        print(
            f"[runtime_proxy] source faces {source_faces} <= target {target_faces}; exporting copy",
            flush=True,
        )
    _delete_loose_vertices(_mesh_objects())
    runtime_topology = _topology_stats(_mesh_objects())
    if runtime_topology["boundary_edges"] > welded_topology["boundary_edges"]:
        raise SystemExit(
            "runtime decimation introduced boundary cracks after position weld: "
            f"{welded_topology['boundary_edges']} -> "
            f"{runtime_topology['boundary_edges']}"
        )
    if args.double_sided:
        _make_materials_double_sided(_mesh_objects())

    actual_faces = _face_count(_mesh_objects())
    actual_vertices = _vertex_count(_mesh_objects())
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB")
    write_runtime_proxy_record(
        metadata_path=metadata,
        source_mesh_path=source,
        runtime_mesh_path=output,
        target_faces=target_faces,
        source_faces=source_faces,
        source_vertices=source_vertices,
        actual_faces=actual_faces,
        actual_vertices=actual_vertices,
        topology={
            "source_import": source_topology,
            "position_weld": weld,
            "source_after_position_weld": welded_topology,
            "runtime_after_decimate": runtime_topology,
            "boundary_cracks_introduced": (
                runtime_topology["boundary_edges"]
                - welded_topology["boundary_edges"]
            ),
        },
    )
    print(
        f"RUNTIME_PROXY_DONE output={output} source_faces={source_faces} "
        f"source_vertices={source_vertices} actual_faces={actual_faces} "
        f"actual_vertices={actual_vertices}",
        flush=True,
    )


if __name__ == "__main__":
    main()
