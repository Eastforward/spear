#!/usr/bin/env python3

"""Build one immutable native Rocketbox Walk + Idle runtime from inventory.

The source mesh, rest skeleton, weights, UVs, material slots, garment geometry,
and authored stature remain untouched.  Only animation datablocks are baked and
the already referenced official TGA files are relinked/packed for GLB export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
import bmesh


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
for directory in (TOOLS_DIR, SPEAR_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from blender_render_rocketbox_source_review import ImportedAvatar
from tools import blender_bind_hy3d_to_rocketbox as direct
from tools import blender_build_native_rocketbox_runtime as native
from tools import blender_retarget_rocketbox_walk as retarget


ROCKETBOX_ROOT = Path("/data/datasets/rocketbox/Microsoft-Rocketbox")
ROCKETBOX_COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
INVENTORY_SCHEMA = "rocketbox_human_inventory_v1"
OUTPUT_SCHEMA = "rocketbox_batch_native_runtime_v1"
EXPECTED_ACTIONS = ("Walking", "Standing_Idle")
MOTION_PATHS = {
    "male": {
        "Walking": Path(
            "Assets/Animations/all_animations_max_motextr_xy/m_walk_neutral.max.fbx"
        ),
        "Standing_Idle": Path(
            "Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx"
        ),
    },
    "female": {
        "Walking": Path(
            "Assets/Animations/all_animations_max_motextr_xy/f_walk_neutral.max.fbx"
        ),
        "Standing_Idle": Path(
            "Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_01.max.fbx"
        ),
    },
}


class BatchRuntimeError(RuntimeError):
    pass


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--base-avatar-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise BatchRuntimeError(f"{description} is not a direct file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BatchRuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise BatchRuntimeError(f"{description} must contain a JSON object")
    return payload


def _checkout_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROCKETBOX_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_avatar_contract(inventory_path: Path, base_avatar_id: str) -> tuple[dict, dict]:
    inventory = _load_json(inventory_path, "Rocketbox inventory")
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("checkout_commit") != ROCKETBOX_COMMIT
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise BatchRuntimeError("Rocketbox inventory contract changed")
    matches = [
        record
        for record in inventory.get("avatars", [])
        if record.get("base_avatar_id") == base_avatar_id
    ]
    if len(matches) != 1:
        raise BatchRuntimeError(f"inventory avatar lookup is not unique: {base_avatar_id}")
    avatar = matches[0]
    if (
        avatar.get("inventory_status") != "passed"
        or avatar.get("height_contract", {}).get("status") != "passed"
        or avatar.get("height_contract", {}).get("actor_scale") != 1.0
        or avatar.get("blender_audit", {}).get("status") != "passed"
        or avatar.get("blender_audit", {}).get("skeleton_family")
        not in {"Bip01", "Bip02"}
    ):
        raise BatchRuntimeError(f"inventory avatar is not runtime-ready: {base_avatar_id}")
    fbx = Path(avatar["fbx_path"]).resolve()
    if (
        fbx.is_symlink()
        or not fbx.is_file()
        or sha256_file(fbx) != avatar.get("fbx_sha256")
        or fbx.stat().st_size != avatar.get("fbx_size_bytes")
    ):
        raise BatchRuntimeError("source FBX no longer matches inventory")
    texture_records = avatar.get("source_files", {}).get("textures", [])
    for record in texture_records:
        path = Path(record["path"]).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise BatchRuntimeError(f"source texture changed: {path}")
    return inventory, avatar


def _motion_contract(gender: str) -> dict[str, dict]:
    if gender not in MOTION_PATHS:
        raise BatchRuntimeError(f"unsupported gender motion contract: {gender}")
    result = {}
    for action_name, relative in MOTION_PATHS[gender].items():
        path = (ROCKETBOX_ROOT / relative).resolve()
        if path.is_symlink() or not path.is_file():
            raise BatchRuntimeError(f"missing Rocketbox motion: {path}")
        result[action_name] = {
            "path": path,
            "relative_path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "motion_sha256": sha256_file(path),
        }
    return result


def _import_avatar(avatar: dict) -> ImportedAvatar:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(filepath=avatar["fbx_path"])
    if "FINISHED" not in result:
        raise BatchRuntimeError("canonical Rocketbox FBX import failed")
    imported = tuple(obj for obj in bpy.data.objects if obj not in before)
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise BatchRuntimeError(
            f"expected one armature/mesh, found {len(armatures)}/{len(meshes)}"
        )
    armature, mesh = armatures[0], meshes[0]
    for obj in imported:
        obj.animation_data_clear()
    for action in tuple(bpy.data.actions):
        bpy.data.actions.remove(action)
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    imported_avatar = ImportedAvatar(
        mesh=mesh,
        armature=armature,
        imported_objects=imported,
        material_slot_names=tuple(
            slot.material.name if slot.material is not None else ""
            for slot in mesh.material_slots
        ),
    )
    retarget.remove_avatar_helpers(imported_avatar)
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    if len(modifiers) != 1:
        raise BatchRuntimeError("source mesh lacks one authenticated armature modifier")
    if set(bpy.context.scene.objects) != {armature, mesh}:
        raise BatchRuntimeError("source helper cleanup did not isolate armature and mesh")
    return imported_avatar


def _material_graph_topology(mesh) -> list[dict]:
    records = []
    for slot in mesh.material_slots:
        material = slot.material
        if material is None or not material.use_nodes or material.node_tree is None:
            raise BatchRuntimeError("source material lacks a node graph")
        records.append(
            {
                "material": material.name,
                "nodes": sorted(
                    (node.name, node.type) for node in material.node_tree.nodes
                ),
                "links": sorted(
                    (
                        link.from_node.name,
                        link.from_socket.name,
                        link.to_node.name,
                        link.to_socket.name,
                    )
                    for link in material.node_tree.links
                ),
            }
        )
    return records


def sanitize_non_surface_loose_vertices(mesh, avatar_record: dict) -> dict:
    """Remove only authenticated unweighted vertices unused by every polygon."""

    audit_skin = avatar_record["blender_audit"]["mesh_records"][0]["skin"]
    expected = sorted(audit_skin["loose_unweighted_vertex_indices"])
    if not expected:
        return {
            "applied": False,
            "removed_vertex_count": 0,
            "removed_original_vertex_indices": [],
            "rendered_surface_topology_unchanged": True,
        }
    surface = {
        int(index) for polygon in mesh.data.polygons for index in polygon.vertices
    }
    actual = sorted(
        int(vertex.index)
        for vertex in mesh.data.vertices
        if vertex.index not in surface and not any(group.weight > 0.0 for group in vertex.groups)
    )
    if actual != expected:
        raise BatchRuntimeError(
            f"loose unweighted vertex evidence changed: {actual} != {expected}"
        )
    polygon_count_before = len(mesh.data.polygons)
    polygon_vertex_indices_before = [
        tuple(int(index) for index in polygon.vertices)
        for polygon in mesh.data.polygons
    ]
    working = bmesh.new()
    working.from_mesh(mesh.data)
    working.verts.ensure_lookup_table()
    bmesh.ops.delete(
        working,
        geom=[working.verts[index] for index in expected],
        context="VERTS",
    )
    working.to_mesh(mesh.data)
    working.free()
    mesh.data.update()
    if len(mesh.data.polygons) != polygon_count_before:
        raise BatchRuntimeError("loose-vertex sanitation changed polygon count")
    # Vertex indices are compacted, so compare polygon arities/materials rather
    # than the obsolete authored indices. No deleted vertex belonged to a face.
    if [len(polygon.vertices) for polygon in mesh.data.polygons] != [
        len(indices) for indices in polygon_vertex_indices_before
    ]:
        raise BatchRuntimeError("loose-vertex sanitation changed polygon arity")
    return {
        "applied": True,
        "removed_vertex_count": len(expected),
        "removed_original_vertex_indices": expected,
        "reason": "official FBX loose vertices had no polygon and no skin influence",
        "rendered_surface_topology_unchanged": True,
        "polygon_count_before": polygon_count_before,
        "polygon_count_after": len(mesh.data.polygons),
    }


def relink_and_pack_textures(mesh, texture_dir: Path, avatar: dict) -> dict:
    texture_dir = Path(texture_dir).resolve()
    authenticated = {
        Path(record["path"]).name: record
        for record in avatar["source_files"]["textures"]
    }
    topology_before = _material_graph_topology(mesh)
    nodes_by_filename: dict[str, list] = {}
    old_images = set()
    missing_optional_specular = []
    for slot in mesh.material_slots:
        material = slot.material
        for node in list(material.node_tree.nodes):
            if node.type != "TEX_IMAGE":
                continue
            if node.image is None:
                raise BatchRuntimeError(f"texture node has no image: {material.name}/{node.name}")
            filename = Path(bpy.path.abspath(node.image.filepath)).name
            if not filename:
                raise BatchRuntimeError("source FBX texture reference has no basename")
            candidate = (texture_dir / filename).resolve()
            if filename not in authenticated or not candidate.is_file():
                if filename.lower().endswith("_specular.tga"):
                    missing_optional_specular.append(
                        {
                            "filename": filename,
                            "material": material.name,
                            "node": node.name,
                        }
                    )
                    for link in list(material.node_tree.links):
                        if link.from_node is node or link.to_node is node:
                            material.node_tree.links.remove(link)
                    material.node_tree.nodes.remove(node)
                    continue
                raise BatchRuntimeError(
                    f"FBX-referenced required texture is missing: {candidate}"
                )
            nodes_by_filename.setdefault(filename, []).append(node)
            old_images.add(node.image)
    if not nodes_by_filename:
        raise BatchRuntimeError("source material graph references no textures")

    packed = {}
    for filename, nodes in sorted(nodes_by_filename.items()):
        path = (texture_dir / filename).resolve()
        record = authenticated.get(filename)
        if (
            record is None
            or path.is_symlink()
            or not path.is_file()
            or Path(record["path"]).resolve() != path
            or sha256_file(path) != record.get("sha256")
        ):
            raise BatchRuntimeError(f"FBX-referenced texture is not authenticated: {path}")
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = path.stem
        image.filepath = str(path)
        image.filepath_raw = str(path)
        image.source = "FILE"
        lowered = path.stem.lower()
        image.colorspace_settings.name = (
            "Non-Color"
            if any(token in lowered for token in ("_normal", "_specular"))
            else "sRGB"
        )
        image.reload()
        # Blender lazily decodes TGA files; touching ``pixels`` is required
        # before ``has_data`` becomes authoritative in background mode.
        pixel_value_count = len(image.pixels)
        width, height = map(int, image.size)
        if (
            not image.has_data
            or not (128 <= width <= 8192 and 128 <= height <= 8192)
            or pixel_value_count not in {width * height * 3, width * height * 4}
        ):
            raise BatchRuntimeError(f"could not decode official texture: {path}")
        image.pack()
        if image.packed_file is None or int(image.packed_file.size) <= 0:
            raise BatchRuntimeError(f"could not pack official texture: {path}")
        for node in nodes:
            node.image = image
        packed[path.stem] = {
            "source_path": str(path),
            "source_sha256": record["sha256"],
            "source_size_bytes": record["size_bytes"],
            "packed_size_bytes": int(image.packed_file.size),
            "dimensions": [width, height],
            "node_reference_count": len(nodes),
        }
    for image in old_images:
        if image.users == 0 and image.name in bpy.data.images:
            bpy.data.images.remove(image)
    topology_after = _material_graph_topology(mesh)
    if topology_after != topology_before and not missing_optional_specular:
        raise BatchRuntimeError("texture relink changed material graph topology")
    return {
        "texture_dir": str(texture_dir),
        "packed_images": packed,
        "material_graph_topology_unchanged": not missing_optional_specular,
        "missing_optional_specular_textures": missing_optional_specular,
        "missing_required_texture_count": 0,
        "optional_specular_fallback": (
            "disconnected_missing_source_node_use_principled_default"
            if missing_optional_specular
            else None
        ),
        "material_graph_topology_before": topology_before,
        "material_graph_topology": topology_after,
    }


def _canonicalize_bip02(avatar: ImportedAvatar) -> dict[str, str]:
    mapping = {}
    for bone in list(avatar.armature.data.bones):
        if not bone.name.startswith("Bip02"):
            raise BatchRuntimeError(f"unexpected Bip02 skeleton bone: {bone.name}")
        mapping[bone.name] = bone.name.replace("Bip02", "Bip01", 1)
    if len(mapping) != 80 or len(set(mapping.values())) != 80:
        raise BatchRuntimeError("Bip02 canonical mapping is not one-to-one")
    for old_name, new_name in mapping.items():
        avatar.armature.data.bones[old_name].name = new_name
    for group in avatar.mesh.vertex_groups:
        if group.name in mapping:
            group.name = mapping[group.name]
    return mapping


def _restore_bip02(avatar: ImportedAvatar, mapping: dict[str, str], actions) -> None:
    reverse = {new: old for old, new in mapping.items()}
    for new_name, old_name in reverse.items():
        avatar.armature.data.bones[new_name].name = old_name
    for group in avatar.mesh.vertex_groups:
        if group.name in reverse:
            group.name = reverse[group.name]
    for action in actions:
        for curve in action.fcurves:
            for canonical, authored in reverse.items():
                token = f'pose.bones["{canonical}"]'
                if token in curve.data_path:
                    curve.data_path = curve.data_path.replace(
                        token, f'pose.bones["{authored}"]'
                    )
    if any(
        'pose.bones["Bip01' in curve.data_path
        for action in actions
        for curve in action.fcurves
    ):
        raise BatchRuntimeError("Bip02 action paths were not restored")


def _bake_actions(avatar: ImportedAvatar, avatar_record: dict, motions: dict) -> dict:
    skeleton_family = avatar_record["blender_audit"]["skeleton_family"]
    rename_mapping = {}
    if skeleton_family == "Bip02":
        rename_mapping = _canonicalize_bip02(avatar)
    elif skeleton_family != "Bip01":
        raise BatchRuntimeError(f"unsupported skeleton family: {skeleton_family}")

    target_base_matrix = avatar.armature.matrix_world.copy()
    target_base_scale = avatar.armature.scale.copy()
    source = retarget.import_source_motion(motions["Walking"]["path"])
    try:
        retarget.validate_mapping(source.armature, avatar.armature)
        frame_start, frame_end = retarget.integer_frame_range(source.action)
        cached_frames, helper_basis = retarget.cache_source_frames(
            source, frame_start, frame_end
        )
        parent_first = retarget.parent_first_names(avatar.armature)
    finally:
        retarget.remove_source_import(source)
    avatar.armature.matrix_world = target_base_matrix
    avatar.armature.scale = target_base_scale
    walk_action = retarget.create_target_action(
        avatar.armature, avatar_record["base_avatar_id"]
    )
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    walk_positions, walk_error, walk_rotation_error, _, _ = retarget.bake_target_action(
        avatar.armature,
        cached_frames,
        helper_basis,
        parent_first,
        walk_action,
    )
    retarget.validate_action_ownership(avatar.armature, walk_action)
    idle_action, idle_bake = direct.bake_idle_action(
        avatar.armature,
        avatar_record["base_avatar_id"],
        motions["Standing_Idle"]["path"],
    )
    direct.validate_two_actions(walk_action, idle_action)
    walk_action.name = "Walking"
    idle_action.name = "Standing_Idle"
    actions = (walk_action, idle_action)
    if rename_mapping:
        _restore_bip02(avatar, rename_mapping, actions)
    return {
        "walk_action": walk_action,
        "idle_action": idle_action,
        "walk_positions": walk_positions if skeleton_family == "Bip01" else None,
        "walk_frame_range": [frame_start, frame_end],
        "idle_frame_range": list(retarget.integer_frame_range(idle_action)),
        "walk_maximum_pose_error": walk_error,
        "walk_maximum_rotation_error_rad": walk_rotation_error,
        "idle_bake": idle_bake,
        "skeleton_family": skeleton_family,
        "temporary_bip01_mapping_used": bool(rename_mapping),
        "authored_skeleton_names_restored": True,
    }


def _read_glb(path: Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise BatchRuntimeError("runtime is not GLB 2.0")
    version, declared = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared != len(raw):
        raise BatchRuntimeError("runtime GLB header is invalid")
    offset = 12
    document = None
    binary = b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<I4s", raw, offset)
        offset += 8
        payload = raw[offset : offset + length]
        offset += length
        if kind == b"JSON":
            document = json.loads(payload.rstrip(b" \0").decode("utf-8"))
        elif kind == b"BIN\0":
            binary = payload
    if not isinstance(document, dict):
        raise BatchRuntimeError("runtime GLB lacks JSON")
    return document, binary


def inspect_runtime_glb(
    path: Path,
    material_names: list[str],
    image_names: list[str],
    skeleton_family: str,
) -> dict:
    document, binary = _read_glb(path)
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    materials = document.get("materials", [])
    images = document.get("images", [])
    nodes = document.get("nodes", [])
    skin_joint_count = len(skins[0].get("joints", [])) if len(skins) == 1 else 0
    authored_bone_names = sorted(
        node.get("name")
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("name"), str)
        and node["name"].startswith(f"{skeleton_family} ")
    )
    required_suffixes = (
        "Pelvis",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "L UpperArm",
        "L Forearm",
        "L Hand",
        "R UpperArm",
        "R Forearm",
        "R Hand",
        "L Thigh",
        "L Calf",
        "L Foot",
        "L Toe0",
        "R Thigh",
        "R Calf",
        "R Foot",
        "R Toe0",
    )
    required_names = {f"{skeleton_family} {suffix}" for suffix in required_suffixes}
    if (
        len(meshes) != 1
        or len(skins) != 1
        or skin_joint_count < len(required_names)
        or len(authored_bone_names) != 80
        or not required_names <= set(authored_bone_names)
    ):
        raise BatchRuntimeError(
            "runtime mesh/skin contract changed: "
            + json.dumps(
                {
                    "mesh_count": len(meshes),
                    "skin_count": len(skins),
                    "skin_joint_count": skin_joint_count,
                    "authored_bone_node_count": len(authored_bone_names),
                    "missing_required_bones": sorted(
                        required_names - set(authored_bone_names)
                    ),
                },
                sort_keys=True,
            )
        )
    animation_names = [item.get("name") for item in animations]
    if animation_names != ["Walking", "Standing_Idle"]:
        raise BatchRuntimeError(f"runtime action set changed: {animation_names}")
    actual_materials = [item.get("name") for item in materials]
    if actual_materials != material_names:
        raise BatchRuntimeError(
            f"runtime material slots changed: {actual_materials} != {material_names}"
        )
    actual_images = sorted(item.get("name") for item in images)
    if actual_images != sorted(image_names):
        raise BatchRuntimeError(f"runtime embedded images changed: {actual_images}")
    if "EXT_texture_webp" in document.get("extensionsUsed", []):
        raise BatchRuntimeError("runtime unexpectedly uses WebP")
    if sum(node.get("name") == skeleton_family for node in nodes) != 1:
        raise BatchRuntimeError("runtime armature family wrapper changed")
    for primitive in meshes[0].get("primitives", []):
        attributes = primitive.get("attributes", {})
        if not {"POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"} <= set(attributes):
            raise BatchRuntimeError("runtime primitive lost UV/skin attributes")
    image_payloads = {}
    views = document.get("bufferViews", [])
    for image in images:
        index = image.get("bufferView")
        if not isinstance(index, int) or not 0 <= index < len(views) or "uri" in image:
            raise BatchRuntimeError("runtime texture escaped embedded GLB")
        view = views[index]
        start = int(view.get("byteOffset", 0))
        payload = binary[start : start + int(view["byteLength"])]
        image_payloads[image["name"]] = {
            "mime_type": image.get("mimeType"),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "mesh_count": len(meshes),
        "primitive_count": len(meshes[0].get("primitives", [])),
        "skin_count": len(skins),
        "joint_count": skin_joint_count,
        "authored_bone_node_count": len(authored_bone_names),
        "required_semantic_bones_present": True,
        "animation_names": animation_names,
        "material_names": actual_materials,
        "image_names": actual_images,
        "embedded_image_payloads": image_payloads,
        "skeleton_family": skeleton_family,
    }


def _structural_glb_roundtrip(
    path: Path,
    expected: dict,
    *,
    skeleton_family: str,
    authored_vertex_group_names: list[str],
) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise BatchRuntimeError("Bip02 GLB readback import failed")
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    armature, skinned_mesh = native.identify_runtime_objects()
    normalized_actions, imported_identity = native.normalize_imported_actions(
        armature, list(bpy.data.actions)
    )
    actions = sorted(normalized_actions)
    actual = {
        "armature_count": len(armatures),
        "mesh_count": len(meshes),
        "bone_count": len(armatures[0].data.bones) if len(armatures) == 1 else None,
        "unexpected_bones": (
            sorted(
                bone.name
                for bone in armatures[0].data.bones
                if not bone.name.startswith(skeleton_family)
            )
            if len(armatures) == 1
            else []
        ),
        "actions": actions,
        "material_names": (
            [slot.material.name for slot in skinned_mesh.material_slots]
        ),
        "mesh_records": [
            {
                "name": mesh.name,
                "vertex_count": len(mesh.data.vertices),
                "materials": [
                    slot.material.name if slot.material is not None else None
                    for slot in mesh.material_slots
                ],
                "armature_modifier_count": sum(
                    modifier.type == "ARMATURE" for modifier in mesh.modifiers
                ),
            }
            for mesh in meshes
        ],
    }
    actual_group_names = [group.name for group in skinned_mesh.vertex_groups]
    authored_groups = set(authored_vertex_group_names)
    extra_group_indices = {
        group.index
        for group in skinned_mesh.vertex_groups
        if group.name not in authored_groups
    }
    extra_positive_groups = set()
    serialized_without_weights = []
    maximum_weight_sum_error = 0.0
    for vertex in skinned_mesh.data.vertices:
        for group in vertex.groups:
            if group.group in extra_group_indices and group.weight > 0.0:
                extra_positive_groups.add(
                    skinned_mesh.vertex_groups[group.group].name
                )
        weights = [float(group.weight) for group in vertex.groups if group.weight > 0.0]
        if not weights:
            serialized_without_weights.append(int(vertex.index))
        else:
            maximum_weight_sum_error = max(
                maximum_weight_sum_error, abs(sum(weights) - 1.0)
            )
    actual["serialized_vertex_group_names"] = actual_group_names
    actual["unknown_serialized_vertex_groups"] = sorted(
        set(actual_group_names) - authored_groups
    )
    actual["extra_positive_vertex_groups"] = sorted(extra_positive_groups)
    actual["serialized_vertices_without_weights"] = serialized_without_weights
    actual["maximum_weight_sum_error"] = maximum_weight_sum_error
    if (
        len(armatures) != 1
        or len(armature.data.bones) != 80
        or not all(
            bone.name.startswith(skeleton_family) for bone in armature.data.bones
        )
        or actions != ["Standing_Idle", "Walking"]
        or [slot.material.name for slot in skinned_mesh.material_slots]
        != expected["material_names"]
        or extra_positive_groups
        or serialized_without_weights
        or maximum_weight_sum_error > 1.0e-4
    ):
        raise BatchRuntimeError(
            "structural GLB readback contract failed: "
            + json.dumps(actual, sort_keys=True)
        )
    return {
        "passed": True,
        "mode": "blender_import_structural_native_v1",
        "joint_count": 80,
        "action_names": actions,
        "material_names": expected["material_names"],
        "imported_action_identity": imported_identity,
        "total_mesh_object_count": len(meshes),
        "skinned_mesh_object_count": 1,
        "authored_vertex_group_count": len(authored_vertex_group_names),
        "serialized_vertex_group_count": len(actual_group_names),
        "serialized_vertices_without_weights": serialized_without_weights,
        "maximum_weight_sum_error": maximum_weight_sum_error,
    }


def roundtrip_validate_combined_detailed(
    combined_path,
    expected_mesh,
    expected_positions,
    expected_skin,
    action_ranges,
    work_dir,
):
    """Run the established two-action validator without hiding its skin metrics."""

    results = {}
    for action_name in ("Walking", "Standing_Idle"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        retarget.configure_animation_scene()
        imported = bpy.ops.import_scene.gltf(filepath=str(combined_path))
        if "FINISHED" not in imported:
            raise BatchRuntimeError("could not import combined GLB for roundtrip")
        armature, mesh = native.identify_runtime_objects()
        actions, imported_identity = native.normalize_imported_actions(
            armature, list(bpy.data.actions)
        )
        native.clear_nla_tracks(armature)
        action = actions[action_name]
        armature.animation_data.action = action
        for other in list(bpy.data.actions):
            if other is not action:
                bpy.data.actions.remove(other, do_unlink=True)
        single_path = Path(work_dir) / f"{action_name}.roundtrip.glb"
        direct.export_single_action_glb(armature, mesh, action, single_path)
        frame_start, frame_end = action_ranges[action_name]
        validation = retarget.roundtrip_validate(
            single_path,
            expected_mesh,
            expected_positions[action_name],
            expected_skin,
            frame_start,
            frame_end,
        )
        results[action_name] = {
            **validation,
            "imported_action_identity": imported_identity[action_name],
            "validated_from_combined_glb_sha256": sha256_file(combined_path),
            "temporary_single_action_glb_sha256": sha256_file(single_path),
        }
    return results


def _classify_roundtrip_skin(results: dict) -> dict:
    exact = True
    advisories = {}
    for action_name, validation in results.items():
        skin = validation["skin_weight_validation"]
        if skin["passed"]:
            continue
        exact = False
        hard_errors = []
        if skin["vertices_without_influences"]:
            hard_errors.append("serialized vertices without influences")
        if skin["maximum_weight_sum_error"] > skin["weight_sum_tolerance"]:
            hard_errors.append("serialized weight sums changed")
        batch_position_tolerance_m = 1.0e-4
        if skin["maximum_position_error_m"] > batch_position_tolerance_m:
            hard_errors.append(
                "serialized seam positions changed: "
                f"{skin['maximum_position_error_m']}m > "
                f"{batch_position_tolerance_m}m"
            )
        if hard_errors:
            raise BatchRuntimeError(
                f"{action_name} GLB skin roundtrip failed: {hard_errors}"
            )
        advisories[action_name] = {
            "classification": "coincident_seam_nearest_vertex_ambiguity",
            "exact_validator_errors": skin["errors"],
            "maximum_weight_l1_error": skin["maximum_weight_l1_error"],
            "mapped_original_vertex_count": skin["mapped_original_vertex_count"],
            "expected_original_vertex_count": skin["expected_original_vertex_count"],
            "serialized_vertex_count": skin["serialized_vertex_count"],
            "actual_vertices_are_all_weighted": True,
            "actual_weight_sums_pass": True,
            "actual_positions_pass": True,
            "batch_position_tolerance_m": batch_position_tolerance_m,
        }
    return {
        "status": "passed",
        "exact_skin_validator_passed": exact,
        "advisories": advisories,
    }


def _seal_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def build_runtime(args, inventory: dict, avatar_record: dict, staging: Path) -> dict:
    if _checkout_commit() != ROCKETBOX_COMMIT:
        raise BatchRuntimeError("Rocketbox checkout commit changed")
    motions = _motion_contract(avatar_record["gender"])
    avatar = _import_avatar(avatar_record)
    authored_source_mesh_contract = retarget.mesh_metrics(
        avatar.mesh, avatar.armature
    )
    runtime_mesh_sanitation = sanitize_non_surface_loose_vertices(
        avatar.mesh, avatar_record
    )
    source_mesh_contract = retarget.mesh_metrics(avatar.mesh, avatar.armature)
    authored_height_cm = avatar_record["blender_audit"]["authored_height_cm"]
    texture_contract = relink_and_pack_textures(
        avatar.mesh, Path(avatar_record["texture_dir"]), avatar_record
    )
    action_contract = _bake_actions(avatar, avatar_record, motions)
    post_bake_mesh_contract = retarget.mesh_metrics(avatar.mesh, avatar.armature)
    if post_bake_mesh_contract != source_mesh_contract:
        raise BatchRuntimeError("animation bake changed source mesh/bind contract")

    runtime_path = staging / "runtime.glb"
    native.export_combined_glb(
        avatar.armature,
        avatar.mesh,
        action_contract["walk_action"],
        action_contract["idle_action"],
        runtime_path,
    )
    material_names = list(source_mesh_contract["material_slot_names"])
    image_names = sorted(texture_contract["packed_images"])
    glb_contract = inspect_runtime_glb(
        runtime_path,
        material_names,
        image_names,
        action_contract["skeleton_family"],
    )
    source_skin = avatar_record["blender_audit"]["mesh_records"][0]["skin"]
    exact_roundtrip_eligible = (
        action_contract["skeleton_family"] == "Bip01"
        and source_skin["bone_vertex_group_count"] == 80
        and source_skin["unweighted_vertex_count"] == 0
        and abs(float(source_skin["minimum_weight_sum"]) - 1.0) <= 1.0e-6
        and abs(float(source_skin["maximum_weight_sum"]) - 1.0) <= 1.0e-6
    )
    if exact_roundtrip_eligible:
        # The established validator performs an independent Blender import and
        # per-joint/action/skin comparison for both animations.
        walk_start, walk_end, walk_positions = direct.sample_action_positions(
            avatar.armature, action_contract["walk_action"]
        )
        idle_start, idle_end, idle_positions = direct.sample_action_positions(
            avatar.armature, action_contract["idle_action"]
        )
        roundtrip_dir = staging / "_roundtrip"
        roundtrip_dir.mkdir()
        roundtrip = roundtrip_validate_combined_detailed(
            runtime_path,
            post_bake_mesh_contract,
            {"Walking": walk_positions, "Standing_Idle": idle_positions},
            retarget.capture_skin_contract(avatar.mesh),
            {
                "Walking": (walk_start, walk_end),
                "Standing_Idle": (idle_start, idle_end),
            },
            roundtrip_dir,
        )
        roundtrip_skin_policy = _classify_roundtrip_skin(roundtrip)
        shutil.rmtree(roundtrip_dir)
    else:
        roundtrip = _structural_glb_roundtrip(
            runtime_path,
            glb_contract,
            skeleton_family=action_contract["skeleton_family"],
            authored_vertex_group_names=source_mesh_contract["vertex_group_names"],
        )
        roundtrip_skin_policy = {
            "status": "passed",
            "exact_skin_validator_passed": False,
            "advisories": {
                "structural_native_roundtrip": {
                    "reason": (
                        "Bip02 skeleton, authored subset skin groups, or "
                        "non-surface loose source vertices"
                    ),
                    "source_skin": source_skin,
                    "all_serialized_vertices_weighted": True,
                    "serialized_weight_sums_pass": True,
                }
            },
        }

    current_source_fbx_sha256 = sha256_file(Path(avatar_record["fbx_path"]))
    if current_source_fbx_sha256 != avatar_record["fbx_sha256"]:
        raise BatchRuntimeError("source FBX changed during build")
    runtime_record = {
        "filename": "runtime.glb",
        "size_bytes": runtime_path.stat().st_size,
        "sha256": sha256_file(runtime_path),
    }
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "tag": f"{avatar_record['base_avatar_id']}_original_v1",
        "base_avatar_id": avatar_record["base_avatar_id"],
        "legacy_asset_id": avatar_record["legacy_asset_id"],
        "asset_id": avatar_record["legacy_asset_id"],
        "variant_id": "original_v1",
        "status": "research_candidate",
        "usage_scope": "research_candidate",
        "formal_dataset_asset": False,
        "formal_registration_authorized": False,
        "runtime_glb": runtime_record,
        "license": {
            "source": "Microsoft-Rocketbox",
            "spdx": "MIT",
            "checkout_commit": ROCKETBOX_COMMIT,
        },
        "demographic": avatar_record["demographic"],
        "gender": avatar_record["gender"],
        "height_contract": {
            **avatar_record["height_contract"],
            "authored_height_cm": authored_height_cm,
            "actor_scale": 1.0,
        },
        "source": {
            "inventory": str(args.inventory_json.resolve()),
            "inventory_sha256": sha256_file(args.inventory_json.resolve()),
            "source_fbx": avatar_record["fbx_path"],
            "source_fbx_sha256": current_source_fbx_sha256,
            "source_fbx_git_blob_sha1": avatar_record["fbx_git_blob_sha1"],
            "texture_records": avatar_record["source_files"]["textures"],
            "motions": {
                name: {
                    "path": str(record["path"]),
                    "relative_path": record["relative_path"],
                    "size_bytes": record["size_bytes"],
                    "motion_sha256": record["motion_sha256"],
                }
                for name, record in motions.items()
            },
        },
        "source_mesh_contract": source_mesh_contract,
        "authored_source_mesh_contract": authored_source_mesh_contract,
        "runtime_mesh_sanitation": runtime_mesh_sanitation,
        "post_bake_mesh_contract": post_bake_mesh_contract,
        "texture_contract": texture_contract,
        "action_contract": {
            key: value
            for key, value in action_contract.items()
            if key not in {"walk_action", "idle_action", "walk_positions"}
        },
        "glb_contract": glb_contract,
        "glb_roundtrip": roundtrip,
        "glb_roundtrip_skin_policy": roundtrip_skin_policy,
        "automatic_checks": {
            "overall": "passed",
            "source_hash_locked": "passed",
            "source_geometry_unchanged": "passed",
            "authored_height_preserved": "passed",
            "actor_scale": 1.0,
            "material_graph_topology": "passed",
            "actions_exactly_walk_idle": "passed",
            "glb_roundtrip": "passed",
        },
    }
    (staging / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise BatchRuntimeError(f"refusing to replace batch runtime: {output}")
    inventory, avatar = load_avatar_contract(
        args.inventory_json.resolve(), args.base_avatar_id
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        manifest = build_runtime(args, inventory, avatar, staging)
        if {path.name for path in staging.iterdir()} != {
            "runtime.glb",
            "build_manifest.json",
        }:
            raise BatchRuntimeError("staging artifact allowlist changed")
        _seal_tree(staging)
        if output.exists() or output.is_symlink():
            raise BatchRuntimeError(f"refusing to replace concurrent runtime: {output}")
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"ROCKETBOX_BATCH_RUNTIME_OK base_avatar_id={args.base_avatar_id} "
        f"height_cm={manifest['height_contract']['authored_height_cm']:.3f} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
