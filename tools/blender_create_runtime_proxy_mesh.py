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
    )
    print(
        f"RUNTIME_PROXY_DONE output={output} source_faces={source_faces} "
        f"source_vertices={source_vertices} actual_faces={actual_faces} "
        f"actual_vertices={actual_vertices}",
        flush=True,
    )


if __name__ == "__main__":
    main()
