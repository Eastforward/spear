"""Create a grounded asymmetric static GLB for emitter integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, -1.5, 0.5))
    mesh = bpy.context.active_object
    mesh.name = "AsymmetricStaticFixture"
    mesh.dimensions = (2.0, 3.0, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    material = bpy.data.materials.new("FixtureMaterial")
    material.diffuse_color = (0.2, 0.4, 0.8, 1.0)
    mesh.data.materials.append(material)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )
    print(f"ASYMMETRIC_STATIC_FIXTURE_OK {output}", flush=True)


if __name__ == "__main__":
    main()
