#!/usr/bin/env python3
"""Deterministic semantic-UV material canaries for the pinned male Route-2 mesh."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_tokenrig_human_static_audit as static_audit  # noqa: E402


SOURCE_ROOT = (
    SPEAR_ROOT
    / "tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/"
    "fitted_skeleton_v1/sanitized_weights_v1/static_audit_v1"
)
SOURCE_GLB = SOURCE_ROOT / "bind_pose.glb"
STATIC_QA = SOURCE_ROOT / "static_qa.json"
SOURCE_GLB_SHA256 = "1a85f2d22e6bdac230379bb57f389db7fc4c73a8f7c50f786e353374f89d6785"
STATIC_QA_SHA256 = "31cd5bf745526913d2226efd180ca10b6623db1b34111f02ae4feef6feae8990"
OUTPUT_ROOT = SPEAR_ROOT / "tmp/route2_deterministic_material_canary_v1"
BLENDER_PATH = Path("/data/jzy/blender/blender-4.2.1-linux-x64/blender")
RENDERER_PATH = SPEAR_ROOT / "tools/blender_render_route2_material_canary.py"
MANIFEST_SCHEMA = "route2_deterministic_material_canary_v1"

REGIONS = ("top", "bottom", "shoes", "hair")
LABELS = {"unknown": 0, "top": 1, "bottom": 2, "shoes": 3, "hair": 4}
LABEL_NAMES = {value: name for name, value in LABELS.items()}
CLASSIFICATION_VERSION = "male_pixal_semantic_core_v1"
REQUIRED_SEMANTIC_ROLES = (
    "pelvis",
    "spine",
    "neck",
    "head",
    "left_clavicle",
    "left_upper_arm",
    "right_clavicle",
    "right_upper_arm",
    "left_thigh",
    "left_calf",
    "right_thigh",
    "right_calf",
    "left_foot",
    "left_toe",
    "right_foot",
    "right_toe",
)
REGISTERED_PALETTE = {
    "top": {"cobalt_blue": (36, 88, 207)},
    "bottom": {"warm_beige": (186, 145, 96)},
    "shoes": {"matte_black": (24, 25, 29)},
    "hair": {"copper_brown": (154, 65, 32)},
}
CANARY_ASSIGNMENTS = {
    "top": "cobalt_blue",
    "bottom": "warm_beige",
    "shoes": "matte_black",
    "hair": "copper_brown",
}

GLB_MAGIC = b"glTF"
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942


class MaterialCanaryError(RuntimeError):
    """Raised when deterministic material evidence is incomplete or stale."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_support(
    joints: np.ndarray,
    weights: np.ndarray,
    joint_names: Sequence[str],
    accepted_names: set[str],
) -> np.ndarray:
    names = np.asarray(joint_names, dtype=object)
    if joints.shape != weights.shape or joints.ndim != 2:
        raise MaterialCanaryError("JOINTS_0 and WEIGHTS_0 shapes differ")
    if joints.size and (int(joints.min()) < 0 or int(joints.max()) >= len(names)):
        raise MaterialCanaryError("JOINTS_0 contains an out-of-range skin slot")
    return (weights * np.isin(names[joints], sorted(accepted_names))).sum(axis=1)


def validate_semantic_bones(
    semantic_bones: Mapping[str, Any], joint_names: Sequence[str]
) -> dict[str, str | list[str]]:
    if not isinstance(semantic_bones, Mapping):
        raise MaterialCanaryError("semantic bone role mapping is missing")
    missing = sorted(set(REQUIRED_SEMANTIC_ROLES) - set(semantic_bones))
    if missing:
        raise MaterialCanaryError(f"semantic bone role mapping is incomplete: {missing}")
    available = set(joint_names)
    if len(available) != len(joint_names):
        raise MaterialCanaryError("skin joint names are not unique")
    result: dict[str, str | list[str]] = {}
    for role in REQUIRED_SEMANTIC_ROLES:
        value = semantic_bones[role]
        if role == "spine":
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(name, str) or not name for name in value)
            ):
                raise MaterialCanaryError("semantic spine role must be a nonempty name list")
            names = list(value)
            result[role] = names
        else:
            if not isinstance(value, str) or not value:
                raise MaterialCanaryError(f"semantic role {role} must be one joint name")
            names = [value]
            result[role] = value
        unknown = sorted(set(names) - available)
        if unknown:
            raise MaterialCanaryError(
                f"semantic role {role} references joints outside the skin: {unknown}"
            )
    return result


def _role_names(
    semantic_bones: Mapping[str, str | list[str]], roles: Sequence[str]
) -> set[str]:
    names: set[str] = set()
    for role in roles:
        value = semantic_bones[role]
        if isinstance(value, list):
            names.update(value)
        else:
            names.add(value)
    return names


def classify_semantic_triangles(
    positions: np.ndarray,
    triangles: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    joint_names: Sequence[str],
    triangle_rgb: np.ndarray,
    *,
    semantic_bones: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify only conservative geometry/color cores; everything else is unknown."""
    positions = np.asarray(positions, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    triangle_rgb = np.asarray(triangle_rgb, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MaterialCanaryError("positions must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise MaterialCanaryError("triangles must have shape (M, 3)")
    if triangle_rgb.shape != (len(triangles), 3):
        raise MaterialCanaryError("triangle RGB samples have the wrong shape")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(weights).all()
        and np.isfinite(triangle_rgb).all()
    ):
        raise MaterialCanaryError("semantic classifier input is non-finite")
    if triangles.size and (int(triangles.min()) < 0 or int(triangles.max()) >= len(positions)):
        raise MaterialCanaryError("triangle index is out of range")

    roles = validate_semantic_bones(semantic_bones, joint_names)

    top_support = _semantic_support(
        joints,
        weights,
        joint_names,
        _role_names(
            roles,
            (
                "pelvis",
                "spine",
                "left_clavicle",
                "left_upper_arm",
                "right_clavicle",
                "right_upper_arm",
            ),
        ),
    )
    bottom_support = _semantic_support(
        joints,
        weights,
        joint_names,
        _role_names(
            roles,
            (
                "pelvis",
                "left_thigh",
                "left_calf",
                "right_thigh",
                "right_calf",
            ),
        ),
    )
    shoe_support = _semantic_support(
        joints,
        weights,
        joint_names,
        _role_names(
            roles,
            ("left_foot", "left_toe", "right_foot", "right_toe"),
        ),
    )
    head_support = _semantic_support(
        joints,
        weights,
        joint_names,
        _role_names(roles, ("neck", "head")),
    )
    centers = positions[triangles].mean(axis=1)
    mean_support = {
        "top": top_support[triangles].mean(axis=1),
        "bottom": bottom_support[triangles].mean(axis=1),
        "shoes": shoe_support[triangles].mean(axis=1),
        "hair": head_support[triangles].mean(axis=1),
    }
    red, green, blue = triangle_rgb.T
    luminance = triangle_rgb.mean(axis=1)
    saturation = triangle_rgb.max(axis=1) - triangle_rgb.min(axis=1)
    height = centers[:, 1]
    depth = centers[:, 2]

    top = (
        (height >= 0.47)
        & (height <= 0.77)
        & (mean_support["top"] >= 0.55)
        & ((green - red) >= 2.0)
        & ((green - blue) >= 5.0)
        & (luminance >= 12.0)
        & (luminance <= 130.0)
    )
    bottom = (
        (height >= 0.08)
        & (height <= 0.54)
        & (mean_support["bottom"] >= 0.60)
        & (saturation <= 24.0)
        & (luminance >= 18.0)
        & (luminance <= 135.0)
    )
    shoes = (
        (height <= 0.105)
        & (mean_support["shoes"] >= 0.65)
        & (saturation <= 38.0)
        & (luminance >= 55.0)
    )
    brown = (
        ((red - green) >= 8.0)
        & ((green - blue) >= 2.0)
        & ((red - blue) >= 14.0)
        & (red < 155.0)
        & (luminance < 120.0)
    )
    hair = (
        (height >= 0.82)
        & (mean_support["hair"] >= 0.70)
        & brown
        & ((depth <= 0.075) | (height >= 0.865))
    )

    labels = np.full(len(triangles), LABELS["unknown"], dtype=np.uint8)
    # The spatial bands are disjoint except at the shoe/trouser boundary. Shoes
    # take precedence there only with strong foot support and a light shoe texel.
    labels[bottom] = LABELS["bottom"]
    labels[top] = LABELS["top"]
    labels[hair] = LABELS["hair"]
    labels[shoes] = LABELS["shoes"]
    counts = {
        name: int(np.count_nonzero(labels == value))
        for name, value in LABELS.items()
    }
    return labels, {
        "classification_version": CLASSIFICATION_VERSION,
        "semantic_role_contract": list(REQUIRED_SEMANTIC_ROLES),
        "triangle_counts": counts,
        "thresholds": {
            "top": {
                "height": [0.47, 0.77],
                "minimum_support": 0.55,
                "green_minus_red_min": 2,
                "green_minus_blue_min": 5,
                "luminance": [12, 130],
            },
            "bottom": {
                "height": [0.08, 0.54],
                "minimum_support": 0.60,
                "saturation_max": 24,
                "luminance": [18, 135],
            },
            "shoes": {
                "height_max": 0.105,
                "minimum_support": 0.65,
                "saturation_max": 38,
                "luminance_min": 55,
            },
            "hair": {
                "height_min": 0.82,
                "minimum_support": 0.70,
                "brown_rgb_margins": [8, 2, 14],
                "red_max": 154,
                "luminance_max": 119,
                "face_guard": "depth<=0.075 or height>=0.865",
            },
        },
    }


def rasterize_uv_triangles(
    uvs: np.ndarray,
    triangles: np.ndarray,
    selected_triangle_indices: Sequence[int] | np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    uvs = np.asarray(uvs, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    selected = np.asarray(selected_triangle_indices, dtype=np.int64)
    if width <= 1 or height <= 1:
        raise MaterialCanaryError("UV raster dimensions are invalid")
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        raise MaterialCanaryError("UVs must have shape (N, 2)")
    if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= len(triangles)):
        raise MaterialCanaryError("selected triangle index is out of range")
    canvas = np.zeros((height, width), dtype=np.uint8)
    if not selected.size:
        return canvas.astype(bool)
    coordinates = uvs[triangles[selected]].copy()
    coordinates[:, :, 0] *= width - 1
    coordinates[:, :, 1] *= height - 1
    coordinates = np.rint(coordinates).astype(np.int32)
    coordinates[:, :, 0] = np.clip(coordinates[:, :, 0], 0, width - 1)
    coordinates[:, :, 1] = np.clip(coordinates[:, :, 1], 0, height - 1)
    for start in range(0, len(coordinates), 20000):
        polygons = [polygon for polygon in coordinates[start : start + 20000]]
        cv2.fillPoly(canvas, polygons, 255, lineType=cv2.LINE_8)
    return canvas > 0


def resolve_mask_conflicts(
    masks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if set(masks) != set(REGIONS):
        raise MaterialCanaryError("semantic mask set is incomplete")
    shapes = {np.asarray(mask).shape for mask in masks.values()}
    if len(shapes) != 1:
        raise MaterialCanaryError("semantic masks have different shapes")
    stack = np.stack([np.asarray(masks[name], dtype=bool) for name in REGIONS])
    conflict = stack.sum(axis=0) > 1
    resolved = {name: stack[index] & ~conflict for index, name in enumerate(REGIONS)}
    return resolved, conflict


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float64) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(srgb * 255.0).astype(np.uint8)


def apply_registered_color(
    rgba: np.ndarray,
    mask: np.ndarray,
    *,
    region: str,
    palette_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgba = np.asarray(rgba, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or mask.shape != rgba.shape[:2]:
        raise MaterialCanaryError("RGBA image and semantic mask shapes differ")
    if region not in REGISTERED_PALETTE or palette_name not in REGISTERED_PALETTE[region]:
        raise MaterialCanaryError("requested region/palette is not a registered palette pair")
    if not mask.any():
        raise MaterialCanaryError("registered semantic mask is empty")
    output = rgba.copy()
    source_linear = _srgb_to_linear(rgba[:, :, :3][mask])
    source_luma = source_linear @ np.asarray([0.2126, 0.7152, 0.0722])
    reference_luma = float(np.median(source_luma))
    if not math.isfinite(reference_luma) or reference_luma <= 1.0e-8:
        raise MaterialCanaryError("semantic region has no usable luminance")
    palette = np.asarray(REGISTERED_PALETTE[region][palette_name], dtype=np.uint8)
    palette_linear = _srgb_to_linear(palette.reshape(1, 3))[0]
    scale = np.clip(source_luma / reference_luma, 0.30, 2.25)
    recolored = _linear_to_srgb(palette_linear[None, :] * scale[:, None])
    output[:, :, :3][mask] = recolored
    changed = np.any(output != rgba, axis=2)
    rgb_changed = np.any(output[:, :, :3] != rgba[:, :, :3], axis=2)
    alpha_changed = output[:, :, 3] != rgba[:, :, 3]
    proof = {
        "region": region,
        "palette_name": palette_name,
        "palette_srgb": palette.tolist(),
        "mask_texels": int(mask.sum()),
        "inside_mask_changed_texels": int(np.count_nonzero(rgb_changed & mask)),
        "outside_mask_changed_texels": int(np.count_nonzero(changed & ~mask)),
        "alpha_changed_texels": int(np.count_nonzero(alpha_changed)),
        "source_linear_luminance_median": reference_luma,
        "transform": "registered_srgb_linear_luminance_scale_v1",
    }
    if proof["outside_mask_changed_texels"] or proof["alpha_changed_texels"]:
        raise MaterialCanaryError("deterministic transform changed protected texels")
    return output, proof


def buffer_view_payload(document: Mapping[str, Any], binary: bytes, index: int) -> bytes:
    views = document.get("bufferViews")
    if not isinstance(views, list) or not (0 <= index < len(views)):
        raise MaterialCanaryError("bufferView index is invalid")
    view = views[index]
    if not isinstance(view, Mapping) or view.get("buffer", 0) != 0:
        raise MaterialCanaryError("bufferView must reference embedded buffer zero")
    offset = view.get("byteOffset", 0)
    length = view.get("byteLength")
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 0:
        raise MaterialCanaryError("bufferView bounds are invalid")
    payload = binary[offset : offset + length]
    if len(payload) != length:
        raise MaterialCanaryError("bufferView exceeds embedded BIN")
    return payload


def replace_buffer_view_payload(
    document: Mapping[str, Any],
    binary: bytes,
    *,
    view_index: int,
    payload: bytes,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    updated = copy.deepcopy(dict(document))
    views = updated.get("bufferViews")
    if not isinstance(views, list) or not (0 <= view_index < len(views)):
        raise MaterialCanaryError("target bufferView index is invalid")
    target = views[view_index]
    start = target.get("byteOffset", 0)
    old_length = target.get("byteLength")
    if not isinstance(start, int) or not isinstance(old_length, int):
        raise MaterialCanaryError("target bufferView bounds are invalid")
    old_end = start + old_length
    later_offsets = [
        view.get("byteOffset", 0)
        for index, view in enumerate(views)
        if index != view_index and isinstance(view.get("byteOffset", 0), int) and view.get("byteOffset", 0) >= old_end
    ]
    tail_start = min(later_offsets, default=len(binary))
    if tail_start < old_end or old_end > len(binary):
        raise MaterialCanaryError("target bufferView overlaps another payload")
    padded = payload + b"\x00" * ((-len(payload)) % 4)
    new_binary = binary[:start] + padded + binary[tail_start:]
    delta = start + len(padded) - tail_start
    target["byteLength"] = len(payload)
    for index, view in enumerate(views):
        if index != view_index and view.get("byteOffset", 0) >= tail_start:
            view["byteOffset"] = view.get("byteOffset", 0) + delta
    buffers = updated.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise MaterialCanaryError("GLB must contain exactly one embedded buffer")
    buffers[0]["byteLength"] = len(new_binary)
    changed = 0
    for index in range(len(views)):
        if index == view_index:
            continue
        if buffer_view_payload(document, binary, index) != buffer_view_payload(updated, new_binary, index):
            changed += 1
    return updated, new_binary, {
        "target_buffer_view": view_index,
        "old_payload_size_bytes": old_length,
        "new_payload_size_bytes": len(payload),
        "non_target_buffer_views_changed": changed,
    }


def encode_glb(document: Mapping[str, Any], binary: bytes) -> bytes:
    value = copy.deepcopy(dict(document))
    buffers = value.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise MaterialCanaryError("GLB encoder requires one buffer")
    buffers[0]["byteLength"] = len(binary)
    json_bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary_padded = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary_padded)
    return b"".join(
        [
            struct.pack("<4sII", GLB_MAGIC, 2, total),
            struct.pack("<II", len(json_bytes), GLB_JSON_CHUNK),
            json_bytes,
            struct.pack("<II", len(binary_padded), GLB_BIN_CHUNK),
            binary_padded,
        ]
    )


def decode_glb_bytes(raw: bytes) -> tuple[dict[str, Any], bytes]:
    if len(raw) < 28:
        raise MaterialCanaryError("GLB is truncated")
    magic, version, declared = struct.unpack_from("<4sII", raw, 0)
    if magic != GLB_MAGIC or version != 2 or declared != len(raw):
        raise MaterialCanaryError("GLB header is invalid")
    offset = 12
    chunks: dict[int, bytes] = {}
    while offset < len(raw):
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[chunk_type] = raw[offset : offset + length]
        offset += length
    if set(chunks) != {GLB_JSON_CHUNK, GLB_BIN_CHUNK}:
        raise MaterialCanaryError("GLB must contain JSON and BIN chunks")
    document = json.loads(chunks[GLB_JSON_CHUNK].decode("utf-8").rstrip(" \x00"))
    declared_binary = document["buffers"][0]["byteLength"]
    return document, chunks[GLB_BIN_CHUNK][:declared_binary]


def _direct_readonly_file(path: Path, expected_sha256: str, description: str) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise MaterialCanaryError(f"{description} must be a direct regular file")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise MaterialCanaryError(f"{description} must be read-only 0444")
    if sha256_file(path) != expected_sha256:
        raise MaterialCanaryError(f"{description} hash changed")
    return path


def _texture_image_index(document: Mapping[str, Any], texture_index: int) -> int:
    textures = document.get("textures")
    if not isinstance(textures, list) or not (0 <= texture_index < len(textures)):
        raise MaterialCanaryError("PBR texture index is invalid")
    texture = textures[texture_index]
    extensions = texture.get("extensions", {})
    if "EXT_texture_webp" in extensions:
        source = extensions["EXT_texture_webp"].get("source")
    else:
        source = texture.get("source")
    images = document.get("images")
    if not isinstance(source, int) or not isinstance(images, list) or not (0 <= source < len(images)):
        raise MaterialCanaryError("PBR texture image binding is invalid")
    return source


def _embedded_image(parsed: Any, image_index: int) -> tuple[bytes, int]:
    image = parsed.document["images"][image_index]
    view_index = image.get("bufferView")
    if not isinstance(view_index, int):
        raise MaterialCanaryError("PBR image is not embedded in the GLB")
    payload = buffer_view_payload(parsed.document, parsed.binary, view_index)
    return payload, view_index


def load_authenticated_source() -> dict[str, Any]:
    """Authenticate and decode the exact static-passed male bind-pose source."""
    source_path = _direct_readonly_file(SOURCE_GLB, SOURCE_GLB_SHA256, "source bind GLB")
    qa_path = _direct_readonly_file(STATIC_QA, STATIC_QA_SHA256, "source static QA")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if (
        qa.get("schema") != "tokenrig_human_static_qa_v1"
        or qa.get("asset_id") != "rocketbox_male_adult_01"
        or qa.get("decision") != "automatic_static_checks_passed"
        or qa.get("checks", {}).get("automatic_static_checks") != "passed"
        or qa.get("checks", {}).get("axis_canonicalization", {}).get("canonical_front")
        != "negative-y"
        or qa.get("checks", {}).get("hierarchy", {}).get("bone_count") != 52
    ):
        raise MaterialCanaryError("source static QA contract changed")
    parsed = static_audit.read_glb(source_path)
    document = parsed.document
    meshes = document.get("meshes")
    skins = document.get("skins")
    materials = document.get("materials")
    if (
        not isinstance(meshes, list)
        or len(meshes) != 1
        or len(meshes[0].get("primitives", [])) != 1
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(materials, list)
        or len(materials) != 1
    ):
        raise MaterialCanaryError("source is not one mesh, one primitive, one material, and one skin")
    primitive = meshes[0]["primitives"][0]
    attributes = primitive.get("attributes", {})
    if primitive.get("mode", 4) != 4 or set(attributes) != {
        "POSITION",
        "NORMAL",
        "TEXCOORD_0",
        "JOINTS_0",
        "WEIGHTS_0",
    }:
        raise MaterialCanaryError("source primitive attribute contract changed")
    positions = static_audit._numpy_glb_accessor(
        np, parsed, attributes["POSITION"]
    ).astype(np.float32)
    uvs = static_audit._numpy_glb_accessor(np, parsed, attributes["TEXCOORD_0"]).astype(
        np.float32
    )
    joints = static_audit._numpy_glb_accessor(
        np, parsed, attributes["JOINTS_0"]
    ).astype(np.int32)
    weights = static_audit._numpy_glb_accessor(
        np, parsed, attributes["WEIGHTS_0"]
    ).astype(np.float32)
    triangles = static_audit._numpy_glb_accessor(
        np, parsed, primitive["indices"]
    ).astype(np.int32).reshape(-1, 3)
    skin_joint_nodes = skins[0].get("joints")
    if not isinstance(skin_joint_nodes, list) or len(skin_joint_nodes) != 52:
        raise MaterialCanaryError("source skin joint set changed")
    joint_names = [document["nodes"][node]["name"] for node in skin_joint_nodes]
    semantic_bones = qa.get("checks", {}).get("semantic_mapping", {}).get(
        "semantic_bones"
    )
    validated_semantic_bones = validate_semantic_bones(semantic_bones, joint_names)
    pbr = materials[0].get("pbrMetallicRoughness", {})
    base_binding = pbr.get("baseColorTexture", {})
    metallic_binding = pbr.get("metallicRoughnessTexture", {})
    if not isinstance(base_binding.get("index"), int) or not isinstance(
        metallic_binding.get("index"), int
    ):
        raise MaterialCanaryError("source PBR texture bindings are incomplete")
    base_image_index = _texture_image_index(document, base_binding["index"])
    metallic_image_index = _texture_image_index(document, metallic_binding["index"])
    base_payload, base_view_index = _embedded_image(parsed, base_image_index)
    metallic_payload, metallic_view_index = _embedded_image(parsed, metallic_image_index)
    base_image = np.asarray(Image.open(io.BytesIO(base_payload)).convert("RGBA"))
    metallic_image = Image.open(io.BytesIO(metallic_payload))
    metallic_image.load()
    if base_image.shape != (4096, 4096, 4) or metallic_image.size != (4096, 4096):
        raise MaterialCanaryError("source PBR texture dimensions changed")
    triangle_uv = uvs[triangles].mean(axis=1)
    pixel_x = np.clip(
        np.floor(triangle_uv[:, 0] * base_image.shape[1]).astype(np.int32),
        0,
        base_image.shape[1] - 1,
    )
    pixel_y = np.clip(
        np.floor(triangle_uv[:, 1] * base_image.shape[0]).astype(np.int32),
        0,
        base_image.shape[0] - 1,
    )
    triangle_rgb = base_image[pixel_y, pixel_x, :3]
    return {
        "path": source_path,
        "static_qa_path": qa_path,
        "parsed": parsed,
        "document": document,
        "binary": parsed.binary,
        "positions": positions,
        "uvs": uvs,
        "joints": joints,
        "weights": weights,
        "triangles": triangles,
        "joint_names": joint_names,
        "semantic_bones": validated_semantic_bones,
        "triangle_rgb": triangle_rgb,
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "joint_count": len(joint_names),
        "texture_size": [base_image.shape[1], base_image.shape[0]],
        "base_color_image": base_image,
        "base_color_image_index": base_image_index,
        "base_color_view_index": base_view_index,
        "base_color_payload": base_payload,
        "base_color_payload_sha256": sha256_bytes(base_payload),
        "metallic_roughness_image_index": metallic_image_index,
        "metallic_roughness_view_index": metallic_view_index,
        "metallic_roughness_payload": metallic_payload,
        "metallic_roughness_payload_sha256": sha256_bytes(metallic_payload),
    }


def _pixel_color_guards(rgba: np.ndarray) -> dict[str, np.ndarray]:
    rgb = rgba[:, :, :3].astype(np.int16)
    red, green, blue = (rgb[:, :, index] for index in range(3))
    luminance = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    opaque = rgba[:, :, 3] >= 250
    return {
        "top": opaque
        & ((green - red) >= 1)
        & ((green - blue) >= 4)
        & (luminance >= 10)
        & (luminance <= 140),
        "bottom": opaque
        & (saturation <= 28)
        & (luminance >= 16)
        & (luminance <= 145),
        "shoes": opaque & (saturation <= 45) & (luminance >= 48),
        "hair": opaque
        & ((red - green) >= 7)
        & ((green - blue) >= 1)
        & ((red - blue) >= 12)
        & (red < 165)
        & (luminance < 130),
    }


def build_semantic_masks(source: Mapping[str, Any]) -> dict[str, Any]:
    labels, classification = classify_semantic_triangles(
        source["positions"],
        source["triangles"],
        source["joints"],
        source["weights"],
        source["joint_names"],
        source["triangle_rgb"],
        semantic_bones=source["semantic_bones"],
    )
    rgba = np.asarray(source["base_color_image"], dtype=np.uint8)
    height, width = rgba.shape[:2]
    uvs = source["uvs"]
    triangles = source["triangles"]
    guards = _pixel_color_guards(rgba)
    unknown_indices = np.flatnonzero(labels == LABELS["unknown"])
    unknown_coverage = rasterize_uv_triangles(
        uvs, triangles, unknown_indices, width=width, height=height
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    provisional: dict[str, np.ndarray] = {}
    unknown_removed: dict[str, int] = {}
    for region in REGIONS:
        selected = np.flatnonzero(labels == LABELS[region])
        raster = rasterize_uv_triangles(
            uvs, triangles, selected, width=width, height=height
        )
        eroded = cv2.erode(raster.astype(np.uint8), kernel, iterations=1) > 0
        guarded = eroded & guards[region]
        unknown_conflict = guarded & unknown_coverage
        provisional[region] = guarded & ~unknown_coverage
        unknown_removed[region] = int(unknown_conflict.sum())
    masks, conflict = resolve_mask_conflicts(provisional)
    stack = np.stack([masks[region] for region in REGIONS]).astype(np.uint8)
    pixel_guard_failures = {
        region: int(np.count_nonzero(masks[region] & ~guards[region]))
        for region in REGIONS
    }
    return {
        "labels": labels,
        "classification": classification,
        "masks": masks,
        "conflict_mask": conflict,
        "unknown_coverage": unknown_coverage,
        "triangle_indices": {
            region: np.flatnonzero(labels == LABELS[region]).astype(np.uint32)
            for region in REGIONS
        },
        "triangle_counts": {
            region: int(np.count_nonzero(labels == LABELS[region])) for region in REGIONS
        },
        "pixel_counts": {region: int(masks[region].sum()) for region in REGIONS},
        "conflict_pixel_count": int(conflict.sum()),
        "post_conflict_overlap_pixel_count": int(np.count_nonzero(stack.sum(axis=0) > 1)),
        "unknown_uv_conflict_removed_pixels": unknown_removed,
        "pixel_color_guard_failures": pixel_guard_failures,
        "pixel_color_guards": {
            "top": "alpha>=250,g-r>=1,g-b>=4,10<=mean(rgb)<=140",
            "bottom": "alpha>=250,saturation<=28,16<=mean(rgb)<=145",
            "shoes": "alpha>=250,saturation<=45,mean(rgb)>=48",
            "hair": "alpha>=250,r-g>=7,g-b>=1,r-b>=12,r<165,mean(rgb)<130",
        },
        "uv_rasterization": {
            "origin": "upper_left_no_v_flip_gltf_image_space",
            "triangle_fill": "opencv_fillPoly_line8",
            "core_erosion_pixels": 1,
            "unknown_triangle_overlap_policy": "remove",
            "cross_region_overlap_policy": "remove_from_all_regions",
        },
    }


def _encode_lossless_webp(rgba: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(
        stream,
        format="WEBP",
        lossless=True,
        quality=100,
        method=4,
        exact=True,
    )
    payload = stream.getvalue()
    decoded = np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"))
    if not np.array_equal(decoded, rgba):
        difference = np.any(decoded != rgba, axis=2)
        raise MaterialCanaryError(
            f"lossless WebP roundtrip changed {int(difference.sum())} decoded texels"
        )
    return payload


def _document_structural_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "asset",
        "extensionsRequired",
        "extensionsUsed",
        "scene",
        "scenes",
        "nodes",
        "meshes",
        "skins",
        "accessors",
        "materials",
        "textures",
        "images",
        "samplers",
    )
    return {key: document.get(key) for key in keys}


def build_variant_bytes(
    source: Mapping[str, Any],
    mask: np.ndarray,
    *,
    region: str,
    palette_name: str,
) -> dict[str, Any]:
    """Build and read back one single-region, lossless-baseColor GLB variant."""
    intended, transform = apply_registered_color(
        source["base_color_image"],
        mask,
        region=region,
        palette_name=palette_name,
    )
    encoded_texture = _encode_lossless_webp(intended)
    updated_document, updated_binary, replacement = replace_buffer_view_payload(
        source["document"],
        source["binary"],
        view_index=source["base_color_view_index"],
        payload=encoded_texture,
    )
    raw = encode_glb(updated_document, updated_binary)
    decoded_document, decoded_binary = decode_glb_bytes(raw)
    output_base_payload = buffer_view_payload(
        decoded_document, decoded_binary, source["base_color_view_index"]
    )
    decoded_rgba = np.asarray(
        Image.open(io.BytesIO(output_base_payload)).convert("RGBA")
    )
    output_metallic = buffer_view_payload(
        decoded_document, decoded_binary, source["metallic_roughness_view_index"]
    )
    changed_non_target = 0
    non_target_records: dict[str, dict[str, Any]] = {}
    for index in range(len(source["document"]["bufferViews"])):
        if index == source["base_color_view_index"]:
            continue
        before = buffer_view_payload(source["document"], source["binary"], index)
        after = buffer_view_payload(decoded_document, decoded_binary, index)
        if before != after:
            changed_non_target += 1
        non_target_records[str(index)] = {
            "source_sha256": sha256_bytes(before),
            "output_sha256": sha256_bytes(after),
            "unchanged": before == after,
        }
    mesh_skin_uv_unchanged = _document_structural_projection(
        source["document"]
    ) == _document_structural_projection(decoded_document)
    qa = {
        **transform,
        **replacement,
        "non_target_buffer_views_changed": changed_non_target,
        "metallic_roughness_payload_unchanged": (
            output_metallic == source["metallic_roughness_payload"]
        ),
        "mesh_skin_uv_document_unchanged": mesh_skin_uv_unchanged,
        "decoded_output_matches_intended_rgba": np.array_equal(decoded_rgba, intended),
        "source_base_color_payload_sha256": source["base_color_payload_sha256"],
        "output_base_color_payload_sha256": sha256_bytes(output_base_payload),
        "source_decoded_rgba_sha256": sha256_bytes(
            np.asarray(source["base_color_image"], dtype=np.uint8).tobytes()
        ),
        "output_decoded_rgba_sha256": sha256_bytes(decoded_rgba.tobytes()),
        "intended_decoded_rgba_sha256": sha256_bytes(intended.tobytes()),
        "non_target_buffer_views": non_target_records,
    }
    if (
        qa["outside_mask_changed_texels"] != 0
        or qa["alpha_changed_texels"] != 0
        or qa["non_target_buffer_views_changed"] != 0
        or not qa["metallic_roughness_payload_unchanged"]
        or not qa["mesh_skin_uv_document_unchanged"]
        or not qa["decoded_output_matches_intended_rgba"]
    ):
        raise MaterialCanaryError(f"variant readback QA failed: {region}")
    return {
        "region": region,
        "palette_name": palette_name,
        "glb_bytes": raw,
        "glb_sha256": sha256_bytes(raw),
        "glb_size_bytes": len(raw),
        "base_color_payload": output_base_payload,
        "base_color_rgba": intended,
        "qa": qa,
    }


def _write_bytes_exclusive(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _write_png_exclusive(path: Path, image: Image.Image) -> None:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    _write_bytes_exclusive(path, stream.getvalue())


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    _write_bytes_exclusive(path, stream.getvalue())


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
    }


def _mask_overview(source_rgba: np.ndarray, masks: Mapping[str, np.ndarray]) -> Image.Image:
    base = np.asarray(source_rgba[:, :, :3], dtype=np.float32)
    output = base.copy()
    colors = {
        "top": np.asarray([35, 105, 255], dtype=np.float32),
        "bottom": np.asarray([230, 175, 95], dtype=np.float32),
        "shoes": np.asarray([230, 55, 75], dtype=np.float32),
        "hair": np.asarray([235, 95, 25], dtype=np.float32),
    }
    for region in REGIONS:
        mask = masks[region]
        output[mask] = base[mask] * 0.25 + colors[region] * 0.75
    return Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8), mode="RGB")


def _texture_comparison(
    source_rgba: np.ndarray,
    variant_rgba: np.ndarray,
    mask: np.ndarray,
    *,
    region: str,
) -> Image.Image:
    source = Image.fromarray(source_rgba, mode="RGBA").convert("RGB")
    variant = Image.fromarray(variant_rgba, mode="RGBA").convert("RGB")
    overlay = np.asarray(source, dtype=np.float32).copy()
    tint = np.asarray(REGISTERED_PALETTE[region][CANARY_ASSIGNMENTS[region]], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.2 + tint * 0.8
    overlay_image = Image.fromarray(np.rint(np.clip(overlay, 0, 255)).astype(np.uint8))
    panels = [source, overlay_image, variant]
    labels = ["SOURCE BASECOLOR", f"REGISTERED {region.upper()} MASK", "LOSSLESS VARIANT"]
    size = 900
    bar = 44
    canvas = Image.new("RGB", (size * 3, size + bar), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        panel = panel.resize((size, size), Image.Resampling.LANCZOS)
        canvas.paste(panel, (index * size, bar))
        draw.text((index * size + 12, 14), label, fill=(235, 240, 248))
    return canvas


def _review_contact(
    source_front: Path,
    render_root: Path,
    *,
    region: str,
) -> Image.Image:
    files = [
        (source_front, "SOURCE FRONT"),
        (render_root / "front.png", "VARIANT FRONT"),
        (render_root / "back.png", "VARIANT BACK"),
        (render_root / "side.png", "VARIANT SIDE"),
    ]
    panel_width, panel_height, bar = 420, 520, 42
    canvas = Image.new("RGB", (panel_width * 4, panel_height + bar), (16, 20, 28))
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(files):
        with Image.open(path) as opened:
            panel = ImageOps.fit(
                opened.convert("RGB"),
                (panel_width, panel_height),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(panel, (index * panel_width, bar))
        draw.text((index * panel_width + 10, 13), label, fill=(240, 244, 250))
    return canvas


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MaterialCanaryError("renameat2 is required for no-replace publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        value = ctypes.get_errno()
        if value == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(value, os.strerror(value), destination)


def _make_readonly_tree(root: Path) -> None:
    for directory, directories, files in os.walk(root, topdown=False, followlinks=False):
        for filename in files:
            path = Path(directory) / filename
            if path.is_symlink():
                raise MaterialCanaryError("published tree contains a symlink")
            path.chmod(0o444)
        for name in directories:
            path = Path(directory) / name
            if path.is_symlink():
                raise MaterialCanaryError("published tree contains a symlink")
            path.chmod(0o555)
    root.chmod(0o555)


def _fsync_tree(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False, followlinks=False):
        for filename in files:
            descriptor = os.open(Path(directory) / filename, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _render_one(
    *,
    staging: Path,
    region: str,
    glb_path: Path,
    glb_sha256: str,
    blender: Path,
    blender_threads: int,
) -> dict[str, Any]:
    output_dir = staging / "renders" / region
    command = [
        str(blender),
        "--threads",
        str(blender_threads),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(RENDERER_PATH),
        "--",
        "--asset-id",
        f"rocketbox_male_adult_01_{region}_color_canary",
        "--input-glb",
        str(glb_path),
        "--input-sha256",
        glb_sha256,
        "--public-relative-glb",
        glb_path.relative_to(staging).as_posix(),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log = (
        "COMMAND "
        + json.dumps(command)
        + "\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr
    ).encode("utf-8")
    log_path = staging / "logs" / f"render_{region}.log"
    _write_bytes_exclusive(log_path, log)
    sentinel = f"ROUTE2_MATERIAL_RENDER_OK asset_id=rocketbox_male_adult_01_{region}_color_canary"
    if completed.returncode != 0 or sentinel not in completed.stdout:
        raise MaterialCanaryError(
            f"Blender material render failed for {region}: returncode={completed.returncode}"
        )
    manifest_path = output_dir / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "route2_deterministic_material_render_v1"
        or manifest.get("canonical_front") != "negative-y"
        or set(manifest.get("views", {})) != {"front", "back", "side"}
        or manifest.get("input_glb", {}).get("sha256") != glb_sha256
    ):
        raise MaterialCanaryError(f"render manifest contract failed for {region}")
    for view in manifest["views"].values():
        path = output_dir / view["filename"]
        if sha256_file(path) != view["sha256"] or path.stat().st_size != view["size_bytes"]:
            raise MaterialCanaryError(f"render artifact changed for {region}")
    return {
        "command": command,
        "manifest": manifest,
        "manifest_record": _relative_record(staging, manifest_path),
        "log_record": _relative_record(staging, log_path),
    }


def _build_staging(
    staging: Path,
    *,
    blender: Path,
    render_workers: int,
    blender_threads: int,
) -> dict[str, Any]:
    source = load_authenticated_source()
    bundle = build_semantic_masks(source)
    masks_root = staging / "masks"
    diagnostics_root = staging / "diagnostics"
    variants_root = staging / "variants"
    for path in (masks_root, diagnostics_root, variants_root, staging / "logs"):
        path.mkdir(parents=True, exist_ok=False)

    _write_npy_exclusive(masks_root / "triangle_labels.npy", bundle["labels"])
    _write_png_exclusive(
        diagnostics_root / "semantic_mask_overlay.png",
        _mask_overview(source["base_color_image"], bundle["masks"]),
    )
    _write_png_exclusive(
        diagnostics_root / "unknown_uv_coverage.png",
        Image.fromarray(bundle["unknown_coverage"].astype(np.uint8) * 255, mode="L"),
    )
    _write_png_exclusive(
        diagnostics_root / "cross_region_conflicts.png",
        Image.fromarray(bundle["conflict_mask"].astype(np.uint8) * 255, mode="L"),
    )
    mask_records: dict[str, Any] = {}
    for region in REGIONS:
        mask_path = masks_root / f"{region}.png"
        triangle_path = masks_root / f"{region}_triangle_indices.npy"
        _write_png_exclusive(
            mask_path,
            Image.fromarray(bundle["masks"][region].astype(np.uint8) * 255, mode="L"),
        )
        _write_npy_exclusive(triangle_path, bundle["triangle_indices"][region])
        mask_records[region] = {
            "mask": _relative_record(staging, mask_path),
            "triangle_indices": _relative_record(staging, triangle_path),
            "triangle_count": bundle["triangle_counts"][region],
            "pixel_count": bundle["pixel_counts"][region],
            "classification_thresholds": bundle["classification"]["thresholds"][region],
            "pixel_color_guard": bundle["pixel_color_guards"][region],
            "unknown_uv_conflict_removed_pixels": bundle[
                "unknown_uv_conflict_removed_pixels"
            ][region],
        }
    registry = {
        "schema": "route2_semantic_uv_mask_registry_v1",
        "asset_id": "rocketbox_male_adult_01",
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "static_qa_sha256": STATIC_QA_SHA256,
        "classification_version": CLASSIFICATION_VERSION,
        "texture_size": source["texture_size"],
        "source_base_color_payload_sha256": source["base_color_payload_sha256"],
        "source_metallic_roughness_payload_sha256": source[
            "metallic_roughness_payload_sha256"
        ],
        "classification": bundle["classification"],
        "uv_rasterization": bundle["uv_rasterization"],
        "conflict_pixel_count": bundle["conflict_pixel_count"],
        "post_conflict_overlap_pixel_count": bundle["post_conflict_overlap_pixel_count"],
        "pixel_color_guard_failures": bundle["pixel_color_guard_failures"],
        "regions": mask_records,
        "status": "registered_research_candidate",
    }
    registry_path = staging / "mask_registry.json"
    _write_json_exclusive(registry_path, registry)

    def make_variant(region: str) -> tuple[str, dict[str, Any]]:
        palette_name = CANARY_ASSIGNMENTS[region]
        result = build_variant_bytes(
            source,
            bundle["masks"][region],
            region=region,
            palette_name=palette_name,
        )
        root = variants_root / region
        root.mkdir(parents=True, exist_ok=False)
        glb_path = root / "canary.glb"
        webp_path = root / "base_color_lossless.webp"
        comparison_path = root / "texture_comparison.png"
        _write_bytes_exclusive(glb_path, result["glb_bytes"])
        _write_bytes_exclusive(webp_path, result["base_color_payload"])
        _write_png_exclusive(
            comparison_path,
            _texture_comparison(
                source["base_color_image"],
                result["base_color_rgba"],
                bundle["masks"][region],
                region=region,
            ),
        )
        return region, {
            "palette_name": palette_name,
            "palette_srgb": list(REGISTERED_PALETTE[region][palette_name]),
            "glb": _relative_record(staging, glb_path),
            "base_color_texture": _relative_record(staging, webp_path),
            "texture_comparison": _relative_record(staging, comparison_path),
            "qa": result["qa"],
            "glb_path": glb_path,
        }

    variants: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(make_variant, region): region for region in REGIONS}
        for future in as_completed(futures):
            region, record = future.result()
            variants[region] = record

    renders: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = {
            executor.submit(
                _render_one,
                staging=staging,
                region=region,
                glb_path=variants[region]["glb_path"],
                glb_sha256=variants[region]["glb"]["sha256"],
                blender=blender,
                blender_threads=blender_threads,
            ): region
            for region in REGIONS
        }
        for future in as_completed(futures):
            region = futures[future]
            renders[region] = future.result()

    source_front = SOURCE_ROOT / "bind_front.png"
    for region in REGIONS:
        contact_path = staging / "renders" / region / "review_contact.png"
        _write_png_exclusive(
            contact_path,
            _review_contact(source_front, staging / "renders" / region, region=region),
        )
        renders[region]["review_contact"] = _relative_record(staging, contact_path)
        variants[region].pop("glb_path")

    if sha256_file(SOURCE_GLB) != SOURCE_GLB_SHA256 or sha256_file(STATIC_QA) != STATIC_QA_SHA256:
        raise MaterialCanaryError("authenticated source changed during canary generation")
    artifacts = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            raise MaterialCanaryError("staging tree contains a symlink")
        if path.is_file():
            artifacts.append(_relative_record(staging, path))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_id": "rocketbox_male_adult_01",
        "classification": "research_candidate",
        "formal_dataset_asset": False,
        "source_choice": (
            "static-passed bind_pose.glb preserves the Pixal PBR surface and adds the authenticated "
            "52-joint skin support required for semantic separation; the original Pixal GLB is unskinned"
        ),
        "source": {
            "bind_pose_glb": _source_record(SOURCE_GLB),
            "static_qa": _source_record(STATIC_QA),
            "canonical_front": "negative-y",
            "vertex_count": source["vertex_count"],
            "triangle_count": source["triangle_count"],
            "joint_count": source["joint_count"],
            "material_count": 1,
            "base_color_payload_sha256": source["base_color_payload_sha256"],
            "metallic_roughness_payload_sha256": source[
                "metallic_roughness_payload_sha256"
            ],
        },
        "mask_registry": {
            "record": _relative_record(staging, registry_path),
            "regions": mask_records,
            "post_conflict_overlap_pixel_count": 0,
        },
        "registered_palette": {
            region: {
                name: list(value) for name, value in REGISTERED_PALETTE[region].items()
            }
            for region in REGIONS
        },
        "variants": variants,
        "renders": renders,
        "execution": {
            "flux_used": False,
            "pixal_inference_used": False,
            "tokenrig_inference_used": False,
            "network_used": False,
            "render_workers": render_workers,
            "blender_threads_per_worker": blender_threads,
            "renderer_sha256": sha256_file(RENDERER_PATH),
            "generator_sha256": sha256_file(Path(__file__)),
        },
        "automatic_qa": {
            "mask_nonoverlap": True,
            "pixel_color_guard_failures": bundle["pixel_color_guard_failures"],
            "all_variants_non_target_texels_exact": all(
                variants[region]["qa"]["outside_mask_changed_texels"] == 0
                for region in REGIONS
            ),
            "all_variants_alpha_exact": all(
                variants[region]["qa"]["alpha_changed_texels"] == 0
                for region in REGIONS
            ),
            "all_variants_non_basecolor_buffers_exact": all(
                variants[region]["qa"]["non_target_buffer_views_changed"] == 0
                for region in REGIONS
            ),
            "all_three_view_renders_present": True,
            "status": "automatic_canary_checks_passed_pending_agent_visual_inspection",
        },
        "artifacts_before_manifest": artifacts,
        "license_and_provenance": {
            "pixal3d": "research_candidate; current NVIDIA-dependent bake/export stack requires release review",
            "skintokens": "MIT code/checkpoints with ArticulationXL/VRoid Hub/ModelsResource provenance risk",
        },
    }
    _write_json_exclusive(staging / "manifest.json", manifest)
    return manifest


def run_canary(
    *,
    output_root: Path = OUTPUT_ROOT,
    blender: Path = BLENDER_PATH,
    render_workers: int = 4,
    blender_threads: int = 24,
) -> Path:
    output_root = Path(output_root).absolute()
    if output_root != OUTPUT_ROOT.absolute():
        raise MaterialCanaryError("material canary output root is not canonical")
    if output_root.exists():
        raise MaterialCanaryError("material canary output already exists")
    blender = Path(blender).absolute()
    if blender.is_symlink() or not blender.is_file() or not os.access(blender, os.X_OK):
        raise MaterialCanaryError("pinned Blender executable is unavailable")
    if not (1 <= render_workers <= 4) or not (1 <= blender_threads <= 112):
        raise MaterialCanaryError("render parallelism is outside the registered bounds")
    staging = output_root.parent / f".{output_root.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    try:
        _build_staging(
            staging,
            blender=blender,
            render_workers=render_workers,
            blender_threads=blender_threads,
        )
        _fsync_tree(staging)
        _make_readonly_tree(staging)
        _rename_noreplace(staging, output_root)
        parent_descriptor = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException as error:
        if staging.exists():
            try:
                _write_json_exclusive(
                    staging / "failure.json",
                    {
                        "schema": "route2_deterministic_material_canary_failure_v1",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "classification": "rejected",
                        "formal_dataset_asset": False,
                    },
                )
            except Exception:
                pass
            failed = output_root.parent / f"{output_root.name}.failed.{uuid.uuid4().hex}"
            _make_readonly_tree(staging)
            _rename_noreplace(staging, failed)
        raise
    return output_root / "manifest.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, default=BLENDER_PATH)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--blender-threads", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_canary(
        blender=args.blender,
        render_workers=args.render_workers,
        blender_threads=args.blender_threads,
    )
    print(
        "ROUTE2_DETERMINISTIC_MATERIAL_CANARY_OK "
        f"manifest={manifest} sha256={sha256_file(manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
