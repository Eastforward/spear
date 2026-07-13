#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Build an immutable native Rocketbox Walking + Standing Idle runtime GLB.

The builder consumes only the sealed approved male Walk blend and files from
the pinned full Microsoft-Rocketbox checkout.  Blender always opens a verified
staging copy; the sealed blend is never saved.  With no arguments it preserves
all seven official texture payloads.  The optional paired variant arguments
replace only ``m002_body_color`` in the same native two-action scene.
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


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_bind_hy3d_to_rocketbox as direct
from tools import blender_retarget_rocketbox_walk as retarget


ASSET_ID = "rocketbox_male_adult_01"
TEXTURE_PREFIX = "m002"
BASELINE_MANIFEST_SHA256 = (
    "b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922"
)
BASELINE_BLEND_SHA256 = (
    "951859fec42091e2e71cc99536996bd29a5536b6e8262f0576fb2b3459fbe603"
)
BASELINE_GLB_SHA256 = (
    "a37cfa20f98ad0308511f6d59ad4571b3e14fa4ffd4ae67b554f1e9e7ee78271"
)
ROCKETBOX_COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
IDLE_SHA256 = "818cc185af21390575f7fbfdeb3012ba2ce5969fbcb220ea725a2617b339a6e2"
IDLE_GIT_BLOB_SHA1 = "a2d92c3326a9c503af677c9fa6082387f060d6c4"
IDLE_SIZE_BYTES = 2418544

BASELINE_ROOT = Path(
    "/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1"
)
BASELINE_MANIFEST = BASELINE_ROOT / "baseline_manifest.json"
BASELINE_ASSET_DIR = BASELINE_ROOT / ASSET_ID
BASELINE_BLEND = BASELINE_ASSET_DIR / "retarget.blend"
BASELINE_GLB = BASELINE_ASSET_DIR / "retarget.glb"
ROCKETBOX_ROOT = Path("/data/datasets/rocketbox/Microsoft-Rocketbox")
IDLE_RELATIVE_PATH = Path(
    "Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx"
)
IDLE_PATH = ROCKETBOX_ROOT / IDLE_RELATIVE_PATH
TEXTURE_DIR = ROCKETBOX_ROOT / "Assets/Avatars/Adults/Male_Adult_01/Textures"
RUNTIME_ROOT = SPEAR_ROOT / "tmp/rocketbox_native_runtime_v1"
ORIGINAL_TAG = "rocketbox_male_adult_01_original_v1"
BLUE_SHIRT_TAG = "rocketbox_male_adult_01_shirt_blue_v1"
ORIGINAL_OUTPUT_DIR = (
    SPEAR_ROOT
    / "tmp/rocketbox_native_runtime_v1/rocketbox_male_adult_01_original_v1"
)

OFFICIAL_TEXTURES = {
    "m002_body_color": {
        "filename": "m002_body_color.tga",
        "size_bytes": 12582956,
        "sha256": "6a048a6b2140a1f5293798ca286be64fd0de6d79572e1273ff5765bd578f463f",
        "git_blob_sha1": "818ec72b6c69655c5853ab0eb1efdd6ab40b2bf0",
    },
    "m002_body_normal": {
        "filename": "m002_body_normal.tga",
        "size_bytes": 12582956,
        "sha256": "c9892e80c56890f6f5365627835286c2d4d2cc34cc473f6788fe3d560a52fb69",
        "git_blob_sha1": "31168ae5dca1085bb433988ff18f37fefe7dba0a",
    },
    "m002_body_specular": {
        "filename": "m002_body_specular.tga",
        "size_bytes": 12582956,
        "sha256": "c071a46bf86f2cc0062ccddae770a9bc21ebae325634439ac3cc1ed4d92fd684",
        "git_blob_sha1": "525c3d89d06bc75b5fd1de4b310f17ee03aff148",
    },
    "m002_head_color": {
        "filename": "m002_head_color.tga",
        "size_bytes": 12582956,
        "sha256": "f5b64e4894930d438f7c419c3c9d0bba1f95fa4de9af078b762ea55bfca1ab85",
        "git_blob_sha1": "0becf592ac705b4a2be8a63aa164266ac2a09d9c",
    },
    "m002_head_normal": {
        "filename": "m002_head_normal.tga",
        "size_bytes": 12582956,
        "sha256": "5ef5ae404c9276f17641e392d0d4bdbe5fb23e94f28238b23a381e6a357c42ea",
        "git_blob_sha1": "9d2385bd43ec4e2bbc1712168d5317d06f9941e7",
    },
    "m002_head_specular": {
        "filename": "m002_head_specular.tga",
        "size_bytes": 12582956,
        "sha256": "c051336b12d443cbd0e04dc3f1ce4109d420f426198aa02a1a1451b87e9a7831",
        "git_blob_sha1": "52b700ec4766105fad73bfc8415036d5db26d2a4",
    },
    "m002_opacity_color": {
        "filename": "m002_opacity_color.tga",
        "size_bytes": 16777260,
        "sha256": "53818d3f45519451edd2bc60d6e59cd4057ecd381827463c5503da0439d0a5cf",
        "git_blob_sha1": "a4088df07a8896761bd209d304b12310f90e5003",
    },
}

EXPECTED_MATERIALS = ("m002_body", "m002_head", "m002_opacity")
EXPECTED_IMAGES = tuple(OFFICIAL_TEXTURES)
EXPECTED_ACTIONS = ("Walking", "Standing_Idle")
EXPECTED_WALK_RANGE = (1, 33)
EXPECTED_IDLE_RANGE = (1, 351)
VARIANT_SCHEMA = "rocketbox_native_body_color_variant_v1"
ORIGINAL_OUTPUT_SCHEMA = "rocketbox_native_runtime_build_v1"
VARIANT_OUTPUT_SCHEMA = "rocketbox_native_material_variant_v1"
ORIGINAL_OUTPUT_MANIFEST = "build_manifest.json"
VARIANT_OUTPUT_MANIFEST = "variant_manifest.json"
OUTPUT_GLB = "runtime.glb"


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-color-texture", type=Path)
    parser.add_argument("--variant-manifest", type=Path)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_regular_file(path, description):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"{description} must be a direct regular file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{description} is empty: {path}")
    return path


def load_json(path, description):
    path = require_regular_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object")
    return payload


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_git(*arguments):
    completed = subprocess.run(
        ["git", "-C", str(ROCKETBOX_ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def file_record(path, git_blob_sha1=None):
    path = require_regular_file(path, path.name)
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if git_blob_sha1 is not None:
        record["git_blob_sha1"] = git_blob_sha1
    return record


def validate_inputs():
    if BASELINE_ROOT.is_symlink() or not BASELINE_ROOT.is_dir():
        raise RuntimeError(f"sealed baseline root is unavailable: {BASELINE_ROOT}")
    if ROCKETBOX_ROOT.is_symlink() or not ROCKETBOX_ROOT.is_dir():
        raise RuntimeError(f"full Rocketbox checkout is unavailable: {ROCKETBOX_ROOT}")

    manifest_path = require_regular_file(BASELINE_MANIFEST, "baseline_manifest.json")
    if sha256_file(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise RuntimeError("sealed baseline manifest hash changed")
    manifest = load_json(manifest_path, "baseline manifest")
    if manifest.get("schema_version") != "rocketbox_baseline_manifest_v1":
        raise RuntimeError("sealed baseline schema changed")
    if manifest.get("baseline_id") != "rocketbox_neutral_walk_v1":
        raise RuntimeError("sealed baseline id changed")
    if manifest.get("motion") != "walk_neutral":
        raise RuntimeError("sealed baseline motion changed")
    asset = manifest.get("assets", {}).get(ASSET_ID)
    if not isinstance(asset, dict):
        raise RuntimeError("sealed baseline is missing the pinned male asset")
    files = asset.get("files", {})

    blend_path = require_regular_file(BASELINE_BLEND, "sealed retarget.blend")
    blend_record = files.get("retarget.blend", {})
    if (
        blend_record.get("sha256") != BASELINE_BLEND_SHA256
        or blend_record.get("size") != blend_path.stat().st_size
        or sha256_file(blend_path) != BASELINE_BLEND_SHA256
    ):
        raise RuntimeError("sealed retarget.blend bytes changed")

    glb_path = require_regular_file(BASELINE_GLB, "sealed retarget.glb")
    glb_record = files.get("retarget.glb", {})
    if (
        glb_record.get("sha256") != BASELINE_GLB_SHA256
        or glb_record.get("size") != glb_path.stat().st_size
        or sha256_file(glb_path) != BASELINE_GLB_SHA256
    ):
        raise RuntimeError("sealed retarget.glb bytes changed")

    checkout_commit = run_git("rev-parse", "HEAD")
    if checkout_commit != ROCKETBOX_COMMIT:
        raise RuntimeError(
            f"Rocketbox checkout commit changed: {checkout_commit} != {ROCKETBOX_COMMIT}"
        )
    idle_path = require_regular_file(IDLE_PATH, "official Rocketbox idle FBX")
    idle_git_blob = run_git("hash-object", str(IDLE_RELATIVE_PATH))
    if (
        idle_path.stat().st_size != IDLE_SIZE_BYTES
        or sha256_file(idle_path) != IDLE_SHA256
        or git_blob_sha1_file(idle_path) != IDLE_GIT_BLOB_SHA1
        or idle_git_blob != IDLE_GIT_BLOB_SHA1
    ):
        raise RuntimeError("official idle bytes do not match the pinned Git blob")

    textures = {}
    for image_name, expected in OFFICIAL_TEXTURES.items():
        relative = TEXTURE_DIR.relative_to(ROCKETBOX_ROOT) / expected["filename"]
        path = require_regular_file(TEXTURE_DIR / expected["filename"], image_name)
        actual_blob = run_git("hash-object", str(relative))
        if (
            path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
            or git_blob_sha1_file(path) != expected["git_blob_sha1"]
            or actual_blob != expected["git_blob_sha1"]
        ):
            raise RuntimeError(f"official texture bytes changed: {image_name}")
        textures[image_name] = file_record(path, actual_blob)

    return {
        "baseline_manifest": file_record(manifest_path),
        "baseline_blend": file_record(blend_path),
        "baseline_glb": file_record(glb_path),
        "checkout_commit": checkout_commit,
        "idle": file_record(idle_path, idle_git_blob),
        "textures": textures,
    }


def validate_variant_request(args):
    body_color_texture = args.body_color_texture
    variant_manifest = args.variant_manifest
    if (body_color_texture is None) != (variant_manifest is None):
        raise RuntimeError(
            "--body-color-texture and --variant-manifest must be supplied together"
        )
    if body_color_texture is None:
        return {
            "mode": "original",
            "variant_id": "original_v1",
            "tag": ORIGINAL_TAG,
            "output_dir": ORIGINAL_OUTPUT_DIR,
            "body_color_texture": None,
            "variant_manifest": None,
        }

    texture_path = require_regular_file(body_color_texture, "body color texture")
    manifest_path = require_regular_file(variant_manifest, "variant manifest")
    manifest = load_json(manifest_path, "variant manifest")
    if manifest.get("schema_version") != "rocketbox_native_body_color_variant_v1":
        raise RuntimeError(f"variant manifest schema must be {VARIANT_SCHEMA}")
    if manifest.get("asset_id") != ASSET_ID:
        raise RuntimeError("variant manifest asset_id must be the pinned male asset")
    if manifest.get("target_image_name") != "m002_body_color":
        raise RuntimeError("variant may target only m002_body_color")
    variant_id = manifest.get("variant_id")
    if not isinstance(variant_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", variant_id) is None:
        raise RuntimeError("variant_id is missing or unsafe")
    if variant_id == "original_v1":
        raise RuntimeError("variant_id may not replace original_v1")
    tag = manifest.get("tag", f"{ASSET_ID}_{variant_id}")
    if tag != f"{ASSET_ID}_{variant_id}" or re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]{0,127}", tag
    ) is None:
        raise RuntimeError("variant tag must be the asset_id plus variant_id")
    expected_sha256 = manifest.get("body_color_texture_sha256")
    expected_size = manifest.get("body_color_texture_size_bytes")
    actual_sha256 = sha256_file(texture_path)
    if expected_sha256 != actual_sha256 or expected_size != texture_path.stat().st_size:
        raise RuntimeError("variant texture bytes do not match its manifest")
    if actual_sha256 == OFFICIAL_TEXTURES["m002_body_color"]["sha256"]:
        raise RuntimeError("variant texture is byte-identical to the original body color")
    return {
        "mode": "body_color_variant",
        "variant_id": variant_id,
        "tag": tag,
        "output_dir": RUNTIME_ROOT / tag,
        "body_color_texture": texture_path,
        "variant_manifest": manifest_path,
        "variant_manifest_payload": manifest,
        "body_color_texture_sha256": actual_sha256,
        "body_color_texture_size_bytes": texture_path.stat().st_size,
        "variant_manifest_sha256": sha256_file(manifest_path),
        "variant_manifest_size_bytes": manifest_path.stat().st_size,
    }


def output_identity(variant):
    if variant["mode"] == "original":
        return {
            "tag": ORIGINAL_TAG,
            "output_dir": ORIGINAL_OUTPUT_DIR,
            "manifest_filename": "build_manifest.json",
            "schema": "rocketbox_native_runtime_build_v1",
        }
    return {
        "tag": variant["tag"],
        "output_dir": variant["output_dir"],
        "manifest_filename": "variant_manifest.json",
        "schema": "rocketbox_native_material_variant_v1",
    }


def collect_source_hashes(inputs, variant):
    hashes = {
        "baseline_manifest": sha256_file(inputs["baseline_manifest"]["path"]),
        "baseline_blend": sha256_file(inputs["baseline_blend"]["path"]),
        "baseline_glb": sha256_file(inputs["baseline_glb"]["path"]),
        "idle_motion_fbx": sha256_file(inputs["idle"]["path"]),
        "official_textures": {
            name: sha256_file(record["path"])
            for name, record in sorted(inputs["textures"].items())
        },
    }
    if variant["mode"] != "original":
        hashes["variant_manifest"] = sha256_file(variant["variant_manifest"])
        hashes["body_color_texture"] = sha256_file(variant["body_color_texture"])
    return hashes


def copy_authenticated_file(source, destination, expected_sha256, expected_size):
    source = require_regular_file(source, Path(source).name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"staging destination already exists: {destination}")
    shutil.copyfile(source, destination)
    if (
        destination.stat().st_size != expected_size
        or sha256_file(destination) != expected_sha256
    ):
        raise RuntimeError(f"staging copy verification failed: {destination}")
    return destination


def stage_inputs(staging_dir, inputs, variant):
    staging_dir = Path(staging_dir)
    input_dir = staging_dir / "_inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    staged = {}
    for key in ("baseline_manifest", "baseline_blend", "baseline_glb", "idle"):
        record = inputs[key]
        suffix = {
            "baseline_manifest": "baseline_manifest.json",
            "baseline_blend": "retarget.blend",
            "baseline_glb": "retarget.glb",
            "idle": "m_idle_neutral_01.max.fbx",
        }[key]
        staged[key] = copy_authenticated_file(
            record["path"], input_dir / suffix, record["sha256"], record["size_bytes"]
        )
    texture_dir = input_dir / "textures"
    staged["textures"] = {}
    for image_name, record in sorted(inputs["textures"].items()):
        destination = texture_dir / OFFICIAL_TEXTURES[image_name]["filename"]
        staged["textures"][image_name] = copy_authenticated_file(
            record["path"], destination, record["sha256"], record["size_bytes"]
        )
    if variant["mode"] != "original":
        staged["body_color_texture"] = copy_authenticated_file(
            variant["body_color_texture"],
            input_dir / "variant" / variant["body_color_texture"].name,
            variant["body_color_texture_sha256"],
            variant["body_color_texture_size_bytes"],
        )
        staged["variant_manifest"] = copy_authenticated_file(
            variant["variant_manifest"],
            input_dir / "variant" / "variant_manifest.json",
            variant["variant_manifest_sha256"],
            variant["variant_manifest_size_bytes"],
        )
    staged_hashes = {
        key: sha256_file(path)
        for key, path in staged.items()
        if key != "textures"
    }
    staged_hashes["textures"] = {
        name: sha256_file(path) for name, path in sorted(staged["textures"].items())
    }
    if any(not value for value in staged_hashes.values()):
        raise RuntimeError("staged input SHA-256 inventory is incomplete")
    return staged


def relink_and_pack_official_textures(staged_textures):
    if len(bpy.data.images) != 7:
        raise RuntimeError(f"sealed blend must contain exactly 7 images, found {len(bpy.data.images)}")
    records = {}
    for image_name, expected in OFFICIAL_TEXTURES.items():
        image = bpy.data.images.get(expected["filename"])
        if image is None:
            image = bpy.data.images.get(image_name)
        if image is None:
            raise RuntimeError(f"sealed blend is missing image datablock {image_name}")
        path = Path(staged_textures[image_name])
        image.filepath = str(path)
        image.filepath_raw = str(path)
        image.source = "FILE"
        image.reload()
        if tuple(image.size) != (2048, 2048) or not image.has_data:
            raise RuntimeError(f"official image failed to load at 2048x2048: {image_name}")
        image.pack()
        if image.packed_file is None:
            raise RuntimeError(f"official image was not packed in staging: {image_name}")
        packed_sha256 = hashlib.sha256(image.packed_file.data).hexdigest()
        if packed_sha256 != expected["sha256"]:
            raise RuntimeError(f"packed official image bytes changed: {image_name}")
        records[image_name] = {
            "datablock_name": image.name,
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "packed_size_bytes": int(image.packed_file.size),
            "image_payload_sha256": packed_sha256,
        }
    return records


def matrix_values(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def socket_default_value(socket):
    if not hasattr(socket, "default_value"):
        return None
    value = socket.default_value
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return str(value)


def material_graph_contract(mesh):
    records = []
    for slot in mesh.material_slots:
        material = slot.material
        if material is None or not material.use_nodes or material.node_tree is None:
            raise RuntimeError("native Rocketbox material graph is missing")
        nodes = []
        for node in sorted(material.node_tree.nodes, key=lambda item: item.name):
            nodes.append(
                {
                    "name": node.name,
                    "type": node.type,
                    "image": node.image.name if node.type == "TEX_IMAGE" and node.image else None,
                    "inputs": {
                        socket.name: socket_default_value(socket)
                        for socket in node.inputs
                        if socket_default_value(socket) is not None
                    },
                }
            )
        links = sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in material.node_tree.links
        )
        records.append(
            {
                "name": material.name,
                "surface_render_method": getattr(material, "surface_render_method", None),
                "blend_method": getattr(material, "blend_method", None),
                "nodes": nodes,
                "links": links,
            }
        )
    return {"records": records, "sha256": canonical_json_sha256(records)}


def capture_native_contract(armature, mesh):
    mesh_metrics = retarget.mesh_metrics(mesh, armature)
    skin_contract = retarget.capture_skin_contract(mesh)
    bone_records = [
        {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent else None,
            "rest_matrix": matrix_values(bone.matrix_local),
        }
        for bone in armature.data.bones
    ]
    material_slot_names = tuple(mesh_metrics["material_slot_names"])
    image_payload = {}
    for image_name, expected in OFFICIAL_TEXTURES.items():
        image = bpy.data.images.get(expected["filename"]) or bpy.data.images.get(image_name)
        if image is None or image.packed_file is None:
            raise RuntimeError(f"native image is missing or unpacked: {image_name}")
        image_payload[image_name] = {
            "size_bytes": int(image.packed_file.size),
            "sha256": hashlib.sha256(image.packed_file.data).hexdigest(),
            "dimensions": [int(image.size[0]), int(image.size[1])],
        }
    return {
        "mesh_metrics": mesh_metrics,
        "skin_contract": skin_contract,
        "skin_summary": {
            "vertex_count": len(skin_contract["vertices"]),
            "group_names": list(skin_contract["group_names"]),
            "bind_mesh_sha256": mesh_metrics["bind_mesh_sha256"],
        },
        "bone_count": len(bone_records),
        "bone_names": [record["name"] for record in bone_records],
        "bone_parent_rest_sha256": canonical_json_sha256(bone_records),
        "material_slot_names": list(material_slot_names),
        "material_graph": material_graph_contract(mesh),
        "image_payload": image_payload,
    }


def manifest_native_contract(contract):
    return {
        key: value
        for key, value in contract.items()
        if key != "skin_contract"
    }


def assert_native_contract_unchanged(before, after, allow_body_color_change=False):
    if before["bone_count"] != 80 or after["bone_count"] != 80:
        raise RuntimeError("native contract must retain exactly 80 bones")
    if len(before["material_slot_names"]) != 3 or len(after["material_slot_names"]) != 3:
        raise RuntimeError("native contract must retain exactly 3 materials")
    if set(before["image_payload"]) != set(EXPECTED_IMAGES) or len(after["image_payload"]) != 7:
        raise RuntimeError("native contract must retain exactly 7 official images")
    for key in (
        "mesh_metrics",
        "skin_summary",
        "bone_count",
        "bone_names",
        "bone_parent_rest_sha256",
        "material_slot_names",
        "material_graph",
    ):
        if before[key] != after[key]:
            raise RuntimeError(f"native mesh/skin/rest/material contract changed: {key}")
    if before["mesh_metrics"]["bind_mesh_sha256"] != after["mesh_metrics"]["bind_mesh_sha256"]:
        raise RuntimeError("native bind_mesh_sha256 changed")
    changed_images = sorted(
        name
        for name in EXPECTED_IMAGES
        if before["image_payload"][name] != after["image_payload"][name]
    )
    expected_changes = ["m002_body_color"] if allow_body_color_change else []
    if changed_images != expected_changes:
        raise RuntimeError(
            f"unexpected packed image payload changes: {changed_images} != {expected_changes}"
        )
    return {"passed": True, "changed_images": changed_images}


def replace_body_color_texture(path):
    target = bpy.data.images.get("m002_body_color.tga") or bpy.data.images.get(
        "m002_body_color"
    )
    if target is None:
        raise RuntimeError("staged scene is missing m002_body_color")
    referencing_nodes = [
        (material.name, node.name)
        for material in bpy.data.materials
        if material.use_nodes and material.node_tree is not None
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is target
    ]
    if not referencing_nodes or any(name != "m002_body" for name, _ in referencing_nodes):
        raise RuntimeError(
            f"m002_body_color must be referenced only by m002_body: {referencing_nodes}"
        )
    original_dimensions = tuple(target.size)
    target.filepath = str(path)
    target.filepath_raw = str(path)
    target.source = "FILE"
    target.reload()
    if tuple(target.size) != original_dimensions or not target.has_data:
        raise RuntimeError(
            f"body-color variant dimensions changed: {tuple(target.size)} != {original_dimensions}"
        )
    target.pack()
    if target.packed_file is None:
        raise RuntimeError("body-color variant was not packed")
    packed_sha256 = hashlib.sha256(target.packed_file.data).hexdigest()
    if packed_sha256 != sha256_file(path):
        raise RuntimeError("packed body-color variant bytes differ from its input")
    return {
        "target_image_name": "m002_body_color",
        "referencing_nodes": referencing_nodes,
        "dimensions": list(original_dimensions),
        "packed_size_bytes": int(target.packed_file.size),
        "packed_sha256": packed_sha256,
    }


def clear_nla_tracks(armature):
    armature.animation_data_create()
    animation_data = armature.animation_data
    for track in list(animation_data.nla_tracks):
        animation_data.nla_tracks.remove(track)
    animation_data.action = None


def add_nla_action(armature, action, name):
    action.name = name
    action.use_fake_user = True
    track = armature.animation_data.nla_tracks.new()
    track.name = name
    frame_start, _ = retarget.integer_frame_range(action)
    strip = track.strips.new(name, frame_start, action)
    strip.name = name
    return track


def export_combined_glb(armature, mesh, walk_action, idle_action, path):
    clear_nla_tracks(armature)
    add_nla_action(armature, walk_action, "Walking")
    add_nla_action(armature, idle_action, "Standing_Idle")
    armature.animation_data.action = None
    direct.select_target_only(armature, mesh)
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_extra_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_frame_range=False,
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"combined GLB export failed: {path}")


def read_glb(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise RuntimeError(f"not a GLB file: {path}")
    version, declared_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_length != len(raw):
        raise RuntimeError("GLB header or declared length is invalid")
    offset = 12
    json_payload = None
    binary_payload = b""
    while offset < len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_payload = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary_payload = chunk
    if not isinstance(json_payload, dict) or not binary_payload:
        raise RuntimeError("GLB is missing JSON or BIN chunks")
    return json_payload, binary_payload


def buffer_view_bytes(payload, binary, index):
    view = payload["bufferViews"][index]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("native GLB unexpectedly uses an external buffer")
    start = int(view.get("byteOffset", 0))
    end = start + int(view["byteLength"])
    if start < 0 or end > len(binary):
        raise RuntimeError("GLB bufferView exceeds the BIN chunk")
    return binary[start:end]


def glb_image_payloads(payload, binary):
    records = {}
    for image in payload.get("images", []):
        name = image.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("GLB image is unnamed")
        if "bufferView" not in image:
            raise RuntimeError("native runtime GLB image must be embedded")
        data = buffer_view_bytes(payload, binary, image["bufferView"])
        records[name] = {
            "mime_type": image.get("mimeType"),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return records


def used_accessor_indices(payload):
    used = set()
    for mesh in payload.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            used.update(primitive.get("attributes", {}).values())
            if "indices" in primitive:
                used.add(primitive["indices"])
            for target in primitive.get("targets", []):
                used.update(target.values())
    for skin in payload.get("skins", []):
        if "inverseBindMatrices" in skin:
            used.add(skin["inverseBindMatrices"])
    for animation in payload.get("animations", []):
        for sampler in animation.get("samplers", []):
            used.add(sampler["input"])
            used.add(sampler["output"])
    return sorted(int(index) for index in used)


def mesh_skin_action_contract(payload, binary):
    accessor_indices = used_accessor_indices(payload)
    accessors = payload.get("accessors", [])
    accessor_records = {}
    view_payload_sha256 = {}
    for index in accessor_indices:
        descriptor = dict(accessors[index])
        accessor_records[str(index)] = descriptor
        view_index = descriptor.get("bufferView")
        if view_index is not None:
            view_payload_sha256[str(view_index)] = hashlib.sha256(
                buffer_view_bytes(payload, binary, view_index)
            ).hexdigest()
    contract = {
        "nodes": payload.get("nodes", []),
        "meshes": payload.get("meshes", []),
        "skins": payload.get("skins", []),
        "animations": payload.get("animations", []),
        "accessors": accessor_records,
        "accessor_buffer_view_sha256": view_payload_sha256,
    }
    return {
        "sha256": canonical_json_sha256(contract),
        "used_accessor_indices": accessor_indices,
        "accessor_buffer_view_sha256": view_payload_sha256,
    }


def inspect_combined_glb(path):
    payload, binary = read_glb(path)
    meshes = payload.get("meshes", [])
    skins = payload.get("skins", [])
    materials = payload.get("materials", [])
    animations = payload.get("animations", [])
    nodes = payload.get("nodes", [])
    mesh_count = len(meshes)
    skin_count = len(skins)
    material_count = len(materials)
    image_payloads = glb_image_payloads(payload, binary)
    image_count = len(image_payloads)
    animation_names = [animation.get("name") for animation in animations]
    if mesh_count != 1:
        raise RuntimeError(f"combined GLB mesh_count must be 1, got {mesh_count}")
    if skin_count != 1:
        raise RuntimeError(f"combined GLB skin_count must be 1, got {skin_count}")
    skin_joint_count = len(skins[0].get("joints", []))
    if skin_joint_count != 80:
        raise RuntimeError(f"combined GLB skin_joint_count must be 80, got {skin_joint_count}")
    joint_names = [nodes[index].get("name") for index in skins[0]["joints"]]
    if set(joint_names) != set(retarget.TARGET_BONES):
        raise RuntimeError("combined GLB joints differ from the native 80-bone contract")
    if animation_names != ["Walking", "Standing_Idle"]:
        raise RuntimeError(
            f"combined GLB animation_names changed: {animation_names}"
        )
    if any(not animation.get("channels") for animation in animations):
        raise RuntimeError("combined GLB contains an empty action")
    if material_count != 3 or tuple(material.get("name") for material in materials) != EXPECTED_MATERIALS:
        raise RuntimeError("combined GLB material contract changed")
    if image_count != 7 or set(image_payloads) != set(EXPECTED_IMAGES):
        raise RuntimeError("combined GLB image contract changed")
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            required = {"TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
            if not required.issubset(primitive.get("attributes", {})):
                raise RuntimeError("combined GLB primitive lost UV or skin attributes")
    semantic = retarget.inspect_semantic_material_bindings(payload, TEXTURE_PREFIX)
    if not semantic["passed"]:
        raise RuntimeError(f"combined GLB material binding failed: {semantic['errors']}")
    helper_names = sorted(
        name
        for name in (node.get("name", "") for node in nodes)
        if "Nub" in name
        or name.startswith("MotionExtractionHelper")
        or name.startswith("ExposeTransformHelper")
        or name.startswith("Bip01 Footsteps")
    )
    if helper_names:
        raise RuntimeError(f"combined GLB leaked source helpers: {helper_names}")
    graph = mesh_skin_action_contract(payload, binary)
    return {
        "mesh_count": mesh_count,
        "skin_count": skin_count,
        "skin_joint_count": skin_joint_count,
        "material_count": material_count,
        "material_names": [material.get("name") for material in materials],
        "image_count": image_count,
        "image_payloads": image_payloads,
        "animation_count": len(animations),
        "animation_names": animation_names,
        "animation_channel_counts": {
            name: len(animation.get("channels", []))
            for name, animation in zip(animation_names, animations)
        },
        "mesh_skin_action_contract_sha256": graph["sha256"],
        "accessor_buffer_view_sha256": graph["accessor_buffer_view_sha256"],
        "semantic_material_bindings": semantic,
        "helper_or_nub_names": helper_names,
    }


def assert_original_pixels_match_sealed_glb(staged_baseline_glb, current_contract):
    payload, binary = read_glb(staged_baseline_glb)
    sealed_images = glb_image_payloads(payload, binary)
    current_images = current_contract["image_payloads"]
    if sealed_images != current_images:
        differences = {
            name: {"sealed": sealed_images.get(name), "runtime": current_images.get(name)}
            for name in sorted(set(sealed_images) | set(current_images))
            if sealed_images.get(name) != current_images.get(name)
        }
        raise RuntimeError(
            f"default runtime image payloads differ from sealed original: {differences}"
        )
    return {
        "passed": True,
        "comparison": "exact_embedded_png_payload_sha256",
        "image_payloads": current_images,
    }


def identify_runtime_objects():
    armatures = [
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ]
    if len(armatures) != 1:
        raise RuntimeError(
            f"combined GLB roundtrip must contain one armature, found {len(armatures)}"
        )
    armature = armatures[0]
    skinned_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object is armature
            for modifier in obj.modifiers
        )
    ]
    if len(skinned_meshes) != 1:
        raise RuntimeError(
            "combined GLB roundtrip must contain one skinned runtime mesh, "
            f"found {len(skinned_meshes)}"
        )
    return armature, skinned_meshes[0]


def normalize_imported_actions(armature, imported_actions):
    imported_actions = list(imported_actions)
    if len(imported_actions) != 2:
        raise RuntimeError(
            f"combined GLB roundtrip must contain two actions, found {len(imported_actions)}"
        )
    animation_data = getattr(armature, "animation_data", None)
    if animation_data is None or not any(
        animation_data.action is action for action in imported_actions
    ):
        raise RuntimeError("combined GLB roundtrip has no active imported action")
    tracks = list(animation_data.nla_tracks)
    if len(tracks) != 2 or {track.name for track in tracks} != set(EXPECTED_ACTIONS):
        raise RuntimeError("combined GLB roundtrip did not preserve exact GLB NLA names")
    normalized = {}
    evidence = {}
    used_actions = []
    for action_name in EXPECTED_ACTIONS:
        track = next(track for track in tracks if track.name == action_name)
        strips = list(track.strips)
        if len(strips) != 1 or not any(
            strips[0].action is imported for imported in imported_actions
        ):
            raise RuntimeError(f"{action_name} NLA track does not own one imported action")
        action = strips[0].action
        if any(action is used for used in used_actions):
            raise RuntimeError("two exact NLA tracks unexpectedly share one action")
        used_actions.append(action)
        imported_name = str(action.name)
        action.name = action_name
        if action.name != action_name:
            raise RuntimeError(f"could not normalize imported action {action_name}")
        normalized[action_name] = action
        evidence[action_name] = {
            "schema": "blender_4_2_gltf_imported_action_identity_v1",
            "glb_animation_name": action_name,
            "nla_track_name": str(track.name),
            "imported_action_datablock_name": imported_name,
            "normalized_action_datablock_name": str(action.name),
            "unique_nla_track_and_strip": True,
        }
    if len(used_actions) != len(imported_actions) or any(
        not any(imported is used for used in used_actions)
        for imported in imported_actions
    ):
        raise RuntimeError("combined GLB has an imported action outside exact NLA tracks")
    return normalized, evidence


def roundtrip_validate_combined(
    combined_path,
    expected_mesh,
    expected_positions,
    expected_skin,
    action_ranges,
    work_dir,
):
    results = {}
    for action_name in ("Walking", "Standing_Idle"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        retarget.configure_animation_scene()
        result = bpy.ops.import_scene.gltf(filepath=str(combined_path))
        if "FINISHED" not in result:
            raise RuntimeError("could not import combined GLB for roundtrip")
        armature, mesh = identify_runtime_objects()
        actions, imported_action_identity = normalize_imported_actions(
            armature, list(bpy.data.actions)
        )
        clear_nla_tracks(armature)
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
        if not validation["skin_weight_validation"]["passed"]:
            raise RuntimeError(f"{action_name} roundtrip skin validation failed")
        results[action_name] = {
            **validation,
            "imported_action_identity": imported_action_identity[action_name],
            "validated_from_combined_glb_sha256": sha256_file(combined_path),
            "temporary_single_action_glb_sha256": sha256_file(single_path),
        }
    return results


def assert_variant_matches_original(current_contract):
    original_manifest_path = ORIGINAL_OUTPUT_DIR / "build_manifest.json"
    original_glb_path = ORIGINAL_OUTPUT_DIR / OUTPUT_GLB
    if not original_manifest_path.is_file() or not original_glb_path.is_file():
        raise RuntimeError("ORIGINAL_OUTPUT_DIR must be built before a body-color variant")
    original_manifest = load_json(original_manifest_path, "original native runtime manifest")
    expected_glb_record = original_manifest.get("runtime_glb", {})
    if (
        expected_glb_record.get("sha256") != sha256_file(original_glb_path)
        or expected_glb_record.get("size_bytes") != original_glb_path.stat().st_size
    ):
        raise RuntimeError("original native runtime GLB no longer matches its manifest")
    original_contract = original_manifest.get("glb_contract", {})
    if (
        original_contract.get("mesh_skin_action_contract_sha256")
        != current_contract.get("mesh_skin_action_contract_sha256")
    ):
        raise RuntimeError("variant mesh/skin/actions differ from original")
    original_images = original_contract.get("image_payloads", {})
    current_images = current_contract.get("image_payloads", {})
    other_image_payload_sha256 = {}
    for image_name in EXPECTED_IMAGES:
        if image_name == "m002_body_color":
            continue
        if original_images.get(image_name) != current_images.get(image_name):
            raise RuntimeError(f"variant changed non-target image {image_name}")
        other_image_payload_sha256[image_name] = current_images[image_name]["sha256"]
    if original_images.get("m002_body_color") == current_images.get("m002_body_color"):
        raise RuntimeError("variant body color GLB payload did not change")
    return {
        "passed": True,
        "original_runtime_glb_sha256": sha256_file(original_glb_path),
        "mesh_skin_action_contract_sha256": current_contract[
            "mesh_skin_action_contract_sha256"
        ],
        "other_image_payload_sha256": other_image_payload_sha256,
        "changed_image": "m002_body_color",
        "original_body_image_payload": original_images["m002_body_color"],
        "variant_body_image_payload": current_images["m002_body_color"],
    }


def build_runtime(
    args,
    inputs,
    variant,
    identity,
    source_hashes_before,
    staging_dir,
):
    staged = stage_inputs(staging_dir, inputs, variant)
    staged_blend = staged["baseline_blend"]
    result = bpy.ops.wm.open_mainfile(filepath=str(staged_blend))
    if "FINISHED" not in result:
        raise RuntimeError("could not open staged sealed retarget.blend")
    retarget.configure_animation_scene()
    armature, mesh = direct.identify_target_objects()
    direct.validate_target_only_scene(armature, mesh)
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("sealed approved Walking action is missing")
    walk_action = armature.animation_data.action
    expected_walk_name = f"{ASSET_ID}_walk_neutral_retarget"
    if walk_action.name != expected_walk_name:
        raise RuntimeError(f"sealed Walking action name changed: {walk_action.name}")
    if retarget.integer_frame_range(walk_action) != EXPECTED_WALK_RANGE:
        raise RuntimeError("sealed Walking frame range changed")
    walk_action.use_fake_user = True

    staged_texture_records = relink_and_pack_official_textures(staged["textures"])
    native_before = capture_native_contract(armature, mesh)
    if native_before["material_slot_names"] != list(EXPECTED_MATERIALS):
        raise RuntimeError("sealed native material slots changed")

    idle_action, idle_bake = direct.bake_idle_action(
        armature, ASSET_ID, staged["idle"]
    )
    if retarget.integer_frame_range(idle_action) != EXPECTED_IDLE_RANGE:
        raise RuntimeError("official Standing Idle frame range changed")
    direct.validate_two_actions(walk_action, idle_action)
    walk_action.name = "Walking"
    idle_action.name = "Standing_Idle"
    action_set = direct.validate_two_actions(walk_action, idle_action)

    walk_start, walk_end, walk_positions = direct.sample_action_positions(
        armature, walk_action
    )
    idle_start, idle_end, idle_positions = direct.sample_action_positions(
        armature, idle_action
    )
    action_ranges = {
        "Walking": (walk_start, walk_end),
        "Standing_Idle": (idle_start, idle_end),
    }
    if action_ranges != {
        "Walking": EXPECTED_WALK_RANGE,
        "Standing_Idle": EXPECTED_IDLE_RANGE,
    }:
        raise RuntimeError(f"native action ranges changed: {action_ranges}")

    variant_texture = None
    if variant["mode"] != "original":
        variant_texture = replace_body_color_texture(staged["body_color_texture"])
    native_after = capture_native_contract(armature, mesh)
    native_contract_check = assert_native_contract_unchanged(
        native_before,
        native_after,
        allow_body_color_change=variant["mode"] != "original",
    )

    runtime_glb = Path(staging_dir) / OUTPUT_GLB
    export_combined_glb(armature, mesh, walk_action, idle_action, runtime_glb)
    glb_contract = inspect_combined_glb(runtime_glb)
    if variant["mode"] == "original":
        original_pixel_check = assert_original_pixels_match_sealed_glb(
            staged["baseline_glb"], glb_contract
        )
        variant_equivalence = None
    else:
        original_pixel_check = None
        variant_equivalence = assert_variant_matches_original(glb_contract)

    roundtrip_dir = Path(staging_dir) / "_roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=False)
    glb_roundtrip = roundtrip_validate_combined(
        runtime_glb,
        native_after["mesh_metrics"],
        {
            "Walking": walk_positions,
            "Standing_Idle": idle_positions,
        },
        native_after["skin_contract"],
        action_ranges,
        roundtrip_dir,
    )

    source_hashes_after = collect_source_hashes(inputs, variant)
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("authenticated source inputs changed during native build")

    runtime_record = {
        "filename": OUTPUT_GLB,
        "size_bytes": runtime_glb.stat().st_size,
        "sha256": sha256_file(runtime_glb),
    }
    manifest = {
        "schema": identity["schema"],
        "tag": identity["tag"],
        "asset_id": ASSET_ID,
        "variant_id": variant["variant_id"],
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
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "sealed_baseline_unchanged": True,
        "inputs": {
            "baseline_manifest": inputs["baseline_manifest"],
            "baseline_blend": inputs["baseline_blend"],
            "baseline_glb": inputs["baseline_glb"],
            "idle": inputs["idle"],
            "official_textures": inputs["textures"],
        },
        "staging": {
            "opened_staged_blend": True,
            "saved_blend": False,
            "official_texture_records": staged_texture_records,
        },
        "variant": {
            "mode": variant["mode"],
            "request_manifest_sha256": variant.get("variant_manifest_sha256"),
            "body_color_texture_sha256": variant.get("body_color_texture_sha256"),
            "texture_replacement": variant_texture,
            "equivalence_to_original": variant_equivalence,
        },
        "native_contract_before": manifest_native_contract(native_before),
        "native_contract_after": manifest_native_contract(native_after),
        "native_contract_check": native_contract_check,
        "actions": {
            "Walking": {
                "source_action_name": expected_walk_name,
                "export_action_name": "Walking",
                "frame_start": walk_start,
                "frame_end": walk_end,
                "fps": 30,
                "source": "sealed approved neutral-walk",
            },
            "Standing_Idle": {
                "source_action_name": f"{ASSET_ID}_idle_neutral_01_retarget",
                "export_action_name": "Standing_Idle",
                "frame_start": idle_start,
                "frame_end": idle_end,
                "fps": 30,
                "source": "official Rocketbox m_idle_neutral_01.max.fbx",
                "bake": idle_bake,
            },
            "set": action_set,
        },
        "glb_contract": glb_contract,
        "glb_roundtrip": glb_roundtrip,
        "original_pixel_check": original_pixel_check,
        "output": {
            "runtime_glb": runtime_record,
        },
        "automatic_checks": {
            "overall": "passed",
            "native_contract": "passed",
            "combined_glb": "passed",
            "both_action_roundtrips": "passed",
            "original_pixels": "passed" if original_pixel_check else "not_applicable",
            "variant_equivalence": "passed" if variant_equivalence else "not_applicable",
        },
    }
    atomic_write_json(
        Path(staging_dir) / identity["manifest_filename"], manifest
    )
    shutil.rmtree(Path(staging_dir) / "_inputs")
    shutil.rmtree(roundtrip_dir)
    print(
        f"ROCKETBOX_NATIVE_RUNTIME_OK asset_id={ASSET_ID} "
        f"variant_id={variant['variant_id']} glb={runtime_glb}",
        flush=True,
    )
    return manifest


def publish_staging(staging_dir, output_dir, manifest_filename):
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"native runtime output already exists (no replace): {output_dir}")
    expected = {OUTPUT_GLB, manifest_filename}
    actual = {path.name for path in staging_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(f"staging output allowlist changed: {sorted(actual)}")
    for path in staging_dir.iterdir():
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeError(f"invalid staged runtime artifact: {path}")
    os.replace(staging_dir, output_dir)


def main(argv=None):
    args = parse_args(argv)
    inputs = validate_inputs()
    variant = validate_variant_request(args)
    identity = output_identity(variant)
    output_dir = identity["output_dir"]
    source_hashes_before = collect_source_hashes(inputs, variant)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"native runtime output already exists (no replace): {output_dir}")
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent
        )
    )
    cleanup = True
    try:
        build_runtime(
            args,
            inputs,
            variant,
            identity,
            source_hashes_before,
            staging_dir,
        )
        publish_staging(
            staging_dir, output_dir, identity["manifest_filename"]
        )
        cleanup = False
    finally:
        if cleanup and staging_dir.exists():
            shutil.rmtree(staging_dir)
    print(f"ROCKETBOX_NATIVE_RUNTIME_PUBLISHED output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
