"""Create a deterministic color variant without changing geometry or rigging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import bpy


SCHEMA = "avengine_semantic_material_color_variant_v1"
HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allowed-material", action="append", default=[])
    parser.add_argument(
        "--material-color",
        action="append",
        default=[],
        help="Exact material assignment NAME=#RRGGBB; may be repeated.",
    )
    parser.add_argument("--attribute-json")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srgb_to_linear(channel):
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def parse_assignments(values):
    assignments = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid material assignment: {value!r}")
        name, color = value.split("=", 1)
        if not name or not HEX_COLOR.fullmatch(color):
            raise SystemExit(f"invalid material assignment: {value!r}")
        if name in assignments:
            raise SystemExit(f"duplicate material assignment: {name}")
        srgb = tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
        assignments[name] = {
            "srgb_hex": color.upper(),
            "linear_rgb": tuple(srgb_to_linear(channel) for channel in srgb),
        }
    return assignments


def main():
    args = parse_argv()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    attribute_json = Path(args.attribute_json).resolve() if args.attribute_json else None
    if source.suffix.lower() != ".glb" or not source.is_file():
        raise SystemExit(f"missing or unsupported input: {source}")
    if attribute_json and not attribute_json.is_file():
        raise SystemExit(f"missing attribute JSON: {attribute_json}")
    for path in (output, manifest):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to replace output: {path}")
    assignments = parse_assignments(args.material_color)
    allowlist = set(args.allowed_material)
    if not assignments or not allowlist:
        raise SystemExit("material assignments and an explicit allowlist are required")
    disallowed = sorted(set(assignments) - allowlist)
    if disallowed:
        raise SystemExit(f"material assignments outside allowlist: {disallowed}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit("input must have exactly one armature")
    armature = armatures[0]
    meshes = [
        item
        for item in bpy.data.objects
        if item.type == "MESH"
        and (
            item.parent is armature
            or any(
                modifier.type == "ARMATURE" and modifier.object is armature
                for modifier in item.modifiers
            )
        )
    ]
    materials = {
        material.name: material
        for mesh in meshes
        for material in mesh.data.materials
        if material is not None
    }
    missing = sorted(set(assignments) - set(materials))
    if missing:
        raise SystemExit(f"input is missing requested materials: {missing}")

    changes = []
    for name, color in assignments.items():
        material = materials[name]
        old_diffuse = [float(value) for value in material.diffuse_color]
        rgba = (*color["linear_rgb"], 1.0)
        material.diffuse_color = rgba
        principled_count = 0
        if material.use_nodes and material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                node.inputs["Base Color"].default_value = rgba
                node.inputs["Alpha"].default_value = 1.0
                principled_count += 1
        changes.append(
            {
                "material": name,
                "old_diffuse_linear_rgba": old_diffuse,
                "new_srgb_hex": color["srgb_hex"],
                "new_linear_rgba": list(rgba),
                "principled_nodes_changed": principled_count,
            }
        )

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_extra_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "attribute_json": (
            {
                "path": str(attribute_json),
                "sha256": sha256_file(attribute_json),
            }
            if attribute_json
            else None
        ),
        "allowed_materials": sorted(allowlist),
        "changes": changes,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "geometry_modified": False,
        "skeleton_modified": False,
        "weights_modified": False,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"SEMANTIC_MATERIAL_RECOLOR_OK output={output}")


if __name__ == "__main__":
    main()
