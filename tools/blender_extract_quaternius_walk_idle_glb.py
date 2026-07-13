"""Extract native Walk and Idle actions from a Quaternius FBX into one GLB.

The source mesh, skeleton, skin weights, and native actions remain unchanged.
Only the target armature and its skinned meshes are exported, with two NLA
tracks named ``Walking`` and ``Idle`` for deterministic downstream discovery.
The FBX materials' legacy zero alpha is repaired to opaque; color values and
material-slot semantics are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy


SCHEMA = "avengine_quaternius_native_walk_idle_extract_v1"


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fbx", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--license-id", default="CC0-1.0")
    parser.add_argument("--license-evidence", required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_action_label(action_name: str) -> str:
    return action_name.rsplit("|", 1)[-1].strip()


def choose_native_action(actions, label):
    matches = [
        action
        for action in actions
        if native_action_label(action.name).lower() == label.lower()
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"native action {label!r} did not resolve uniquely; "
            f"available={[native_action_label(action.name) for action in actions]}"
        )
    return matches[0]


def skinned_meshes_for(armature):
    return [
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


def add_track(armature, action, name):
    action.name = name
    action.use_fake_user = True
    track = armature.animation_data.nla_tracks.new()
    track.name = name
    strip = track.strips.new(name, int(round(action.frame_range[0])), action)
    strip.name = name


def repair_legacy_zero_alpha(meshes):
    records = []
    materials = {
        material
        for mesh in meshes
        for material in mesh.data.materials
        if material is not None
    }
    for material in sorted(materials, key=lambda item: item.name):
        old_diffuse_alpha = float(material.diffuse_color[3])
        material.diffuse_color[3] = 1.0
        old_principled_alpha = None
        if material.use_nodes and material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                alpha = node.inputs.get("Alpha")
                base_color = node.inputs.get("Base Color")
                if alpha is not None:
                    old_principled_alpha = float(alpha.default_value)
                    alpha.default_value = 1.0
                if base_color is not None:
                    value = list(base_color.default_value)
                    value[3] = 1.0
                    base_color.default_value = value
        records.append(
            {
                "material": material.name,
                "old_diffuse_alpha": old_diffuse_alpha,
                "old_principled_alpha": old_principled_alpha,
                "new_alpha": 1.0,
            }
        )
    return records


def main():
    args = parse_argv()
    source = Path(args.input_fbx).resolve()
    output = Path(args.output_glb).resolve()
    manifest = Path(args.output_manifest).resolve()
    evidence = Path(args.license_evidence).resolve()
    if source.suffix.lower() != ".fbx" or not source.is_file():
        raise SystemExit(f"missing or unsupported source: {source}")
    if not evidence.is_file():
        raise SystemExit(f"missing license evidence: {evidence}")
    for path in (output, manifest):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to replace output: {path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source), use_anim=True)
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit(f"expected exactly one armature, got {len(armatures)}")
    armature = armatures[0]
    meshes = skinned_meshes_for(armature)
    if not meshes:
        raise SystemExit("source has no skinned mesh")
    source_actions = list(bpy.data.actions)
    walking = choose_native_action(source_actions, "Walk")
    idle = choose_native_action(source_actions, "Idle")
    material_repairs = repair_legacy_zero_alpha(meshes)

    armature.animation_data_create()
    armature.animation_data.action = None
    for track in list(armature.animation_data.nla_tracks):
        armature.animation_data.nla_tracks.remove(track)
    add_track(armature, walking, "Walking")
    add_track(armature, idle, "Idle")
    armature.animation_data.action = None

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
            "armature": armature.name,
            "bone_count": len(armature.data.bones),
            "skinned_meshes": [
                {
                    "name": mesh.name,
                    "vertices": len(mesh.data.vertices),
                    "polygons": len(mesh.data.polygons),
                }
                for mesh in meshes
            ],
            "all_native_actions": [
                native_action_label(action.name) for action in source_actions
            ],
            "selected_native_actions": ["Walk", "Idle"],
        },
        "license": {
            "spdx_or_policy_id": args.license_id,
            "evidence_path": str(evidence),
            "evidence_sha256": sha256_file(evidence),
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "canonical_actions": ["Walking", "Idle"],
            "material_alpha_repairs": material_repairs,
        },
        "status": "research_candidate_pending_deformation_and_media_qa",
        "formal_dataset_registration_authorized": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"QUATERNIUS_WALK_IDLE_EXTRACT_OK output={output}", flush=True)


if __name__ == "__main__":
    main()
