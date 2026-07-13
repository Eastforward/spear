#!/usr/bin/env python3
"""Deterministic native-texture material canary for a pinned Rocketbox avatar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image


SPEAR_ROOT = Path(__file__).resolve().parents[1]
ROCKETBOX_ROOT = Path("/data/datasets/rocketbox/Microsoft-Rocketbox")
ROCKETBOX_COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
ASSET_ID = "rocketbox_male_adult_01"
SOURCE_BODY_COLOR_SHA256 = (
    "6a048a6b2140a1f5293798ca286be64fd0de6d79572e1273ff5765bd578f463f"
)
SOURCE_BODY_COLOR_GIT_BLOB_SHA1 = "818ec72b6c69655c5853ab0eb1efdd6ab40b2bf0"
SOURCE_BODY_COLOR_SIZE = 12_582_956
OUTPUT_ROOT = (
    SPEAR_ROOT
    / "tmp/rocketbox_native_material_canary_v1/"
    "rocketbox_male_adult_01/shirt_blue_v1"
)
BLENDER_PATH = Path("/data/jzy/blender/blender-4.2.1-linux-x64/blender")
MESH_NAME = "m002_hipoly_81_bones_opacity"
SHIRT_ISLAND_SIGNATURES = [
    [80, 0.009, 0.3953, 0.2767, 0.5582, 123.73, 148.34],
    [483, 0.3565, 0.1508, 0.6428, 0.9439, 89.01, 157.6],
    [80, 0.7228, 0.3948, 0.9905, 0.5568, 123.3, 148.54],
]
SHIRT_FACE_COUNT = 643
SHIRT_FACE_INDICES_U32LE_SHA256 = (
    "b6e68ad480a10f9756129ca6360596e5c5a91e641a19ba0fbf9d96d91267f317"
)
STRIPE_LUMA_THRESHOLD = 145
BUTTON_GUARDS = ((968, 952, 18), (979, 1010, 18), (983, 1064, 18))
FROZEN_MASK_REGISTRY = {
    "shirt_surface": {
        "pixel_count": 1_129_206,
        "raw_bool_sha256": "8ac1a005a391da508748b25f93d7e4385bcba50980ae55850bc20d7c447f382f",
    },
    "stripe_detail_protect": {
        "pixel_count": 325_654,
        "raw_bool_sha256": "8ed02cb4f9b5bdf7520787d19d338e15204978b0645685e80966f45979dca294",
    },
    "shirt_main_color": {
        "pixel_count": 803_552,
        "raw_bool_sha256": "0a5459ae32f2b74824be07ce43fb9d6673e851e78505354eafb741f30e7f6ca9",
    },
}
FROZEN_BLUE_TGA_SHA256 = (
    "1098a88989e10574ec5d3c058f51cb17164b5671a47649071ad74ed514ec04e9"
)

SOURCE_FBX_RELATIVE = Path(
    "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx"
)
TEXTURE_ROOT_RELATIVE = Path(
    "Assets/Avatars/Adults/Male_Adult_01/Textures"
)
SOURCE_BODY_COLOR_RELATIVE = TEXTURE_ROOT_RELATIVE / "m002_body_color.tga"
SOURCE_RECORDS = {
    "fbx": {
        "relative_path": SOURCE_FBX_RELATIVE,
        "size_bytes": 521_472,
        "sha256": "8d1edb51b4dc3427ae2456f4407fc105532c145dd019e53cd42bab31cc948a29",
        "git_blob_sha1": "fdaf4aa7d15054f1601740ea1a09cf111938d210",
    },
    "body_color": {
        "relative_path": SOURCE_BODY_COLOR_RELATIVE,
        "size_bytes": SOURCE_BODY_COLOR_SIZE,
        "sha256": SOURCE_BODY_COLOR_SHA256,
        "git_blob_sha1": SOURCE_BODY_COLOR_GIT_BLOB_SHA1,
    },
}
PROTECTED_TEXTURE_SPECS = {
    "body_normal": (
        "m002_body_normal.tga",
        12_582_956,
        "c9892e80c56890f6f5365627835286c2d4d2cc34cc473f6788fe3d560a52fb69",
        "31168ae5dca1085bb433988ff18f37fefe7dba0a",
    ),
    "body_specular": (
        "m002_body_specular.tga",
        12_582_956,
        "c071a46bf86f2cc0062ccddae770a9bc21ebae325634439ac3cc1ed4d92fd684",
        "525c3d89d06bc75b5fd1de4b310f17ee03aff148",
    ),
    "head_color": (
        "m002_head_color.tga",
        12_582_956,
        "f5b64e4894930d438f7c419c3c9d0bba1f95fa4de9af078b762ea55bfca1ab85",
        "0becf592ac705b4a2be8a63aa164266ac2a09d9c",
    ),
    "head_normal": (
        "m002_head_normal.tga",
        12_582_956,
        "5ef5ae404c9276f17641e392d0d4bdbe5fb23e94f28238b23a381e6a357c42ea",
        "9d2385bd43ec4e2bbc1712168d5317d06f9941e7",
    ),
    "head_specular": (
        "m002_head_specular.tga",
        12_582_956,
        "c051336b12d443cbd0e04dc3f1ce4109d420f426198aa02a1a1451b87e9a7831",
        "52b700ec4766105fad73bfc8415036d5db26d2a4",
    ),
    "opacity_color": (
        "m002_opacity_color.tga",
        16_777_260,
        "53818d3f45519451edd2bc60d6e59cd4057ecd381827463c5503da0439d0a5cf",
        "a4088df07a8896761bd209d304b12310f90e5003",
    ),
}
LICENSE_SPEC = {
    "relative_path": Path("LICENSE.md"),
    "size_bytes": 1_066,
    "sha256": "17474e386e0b9e1a700cc3d06b2b0882a2c376d9c6b49c7f8274409b8f8d2352",
    "git_blob_sha1": "9bcfb3ece5301a55d3a41bbd00a029ab27d61d13",
}


class NativeMaterialCanaryError(RuntimeError):
    """Raised when pinned Rocketbox evidence is missing, stale, or unsafe."""


_BLENDER_UV_EXTRACT_CODE = r'''
import bpy
import json
import sys

fbx_path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.import_scene.fbx(filepath=fbx_path)
mesh_object = bpy.data.objects.get("m002_hipoly_81_bones_opacity")
if mesh_object is None or mesh_object.type != "MESH":
    raise RuntimeError("pinned Rocketbox mesh object is missing")
mesh = mesh_object.data
uv_layer = mesh.uv_layers.active
if uv_layer is None:
    raise RuntimeError("pinned Rocketbox mesh has no active UV layer")
uv_data = uv_layer.data
body_polygons = [polygon for polygon in mesh.polygons if polygon.material_index == 0]
parent = {polygon.index: polygon.index for polygon in body_polygons}

def find(index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index

def union(left, right):
    left_root = find(left)
    right_root = find(right)
    if left_root != right_root:
        parent[right_root] = left_root

edge_records = {}
for polygon in body_polygons:
    loops = list(polygon.loop_indices)
    for offset, loop_index in enumerate(loops):
        next_loop_index = loops[(offset + 1) % len(loops)]
        vertex_index = mesh.loops[loop_index].vertex_index
        next_vertex_index = mesh.loops[next_loop_index].vertex_index
        mapping = {
            vertex_index: (
                round(float(uv_data[loop_index].uv.x), 6),
                round(float(uv_data[loop_index].uv.y), 6),
            ),
            next_vertex_index: (
                round(float(uv_data[next_loop_index].uv.x), 6),
                round(float(uv_data[next_loop_index].uv.y), 6),
            ),
        }
        edge_key = tuple(sorted((vertex_index, next_vertex_index)))
        for other_index, other_mapping in edge_records.get(edge_key, []):
            if (
                other_mapping.get(vertex_index) == mapping.get(vertex_index)
                and other_mapping.get(next_vertex_index)
                == mapping.get(next_vertex_index)
            ):
                union(polygon.index, other_index)
        edge_records.setdefault(edge_key, []).append((polygon.index, mapping))

islands = {}
for polygon in body_polygons:
    islands.setdefault(find(polygon.index), []).append(polygon)

expected_signatures = {
    (80, 0.009, 0.3953, 0.2767, 0.5582, 123.73, 148.34),
    (483, 0.3565, 0.1508, 0.6428, 0.9439, 89.01, 157.6),
    (80, 0.7228, 0.3948, 0.9905, 0.5568, 123.3, 148.54),
}
selected = {}
selected_signatures = []
for polygons in islands.values():
    loop_indices = [index for polygon in polygons for index in polygon.loop_indices]
    vertex_indices = sorted(
        {mesh.loops[index].vertex_index for index in loop_indices}
    )
    u_values = [float(uv_data[index].uv.x) for index in loop_indices]
    v_values = [float(uv_data[index].uv.y) for index in loop_indices]
    z_values = [float(mesh.vertices[index].co.z) for index in vertex_indices]
    signature = (
        len(polygons),
        round(min(u_values), 4),
        round(min(v_values), 4),
        round(max(u_values), 4),
        round(max(v_values), 4),
        round(min(z_values), 2),
        round(max(z_values), 2),
    )
    if signature in expected_signatures:
        selected_signatures.append(list(signature))
        for polygon in polygons:
            selected[polygon.index] = [
                [
                    float(uv_data[loop_index].uv.x),
                    float(uv_data[loop_index].uv.y),
                ]
                for loop_index in polygon.loop_indices
            ]

polygon_indices = sorted(selected)
material_names = [
    material.name if material is not None else None for material in mesh.materials
]
material_polygon_counts = [
    sum(1 for polygon in mesh.polygons if polygon.material_index == index)
    for index in range(len(mesh.materials))
]
payload = {
    "mesh_name": mesh_object.name,
    "material_names": material_names,
    "material_polygon_counts": material_polygon_counts,
    "body_uv_island_count": len(islands),
    "shirt_island_signatures": sorted(selected_signatures, key=lambda value: value[1]),
    "polygon_indices": polygon_indices,
    "uv_polygons": [selected[index] for index in polygon_indices],
}
print("ROCKETBOX_NATIVE_UV_JSON=" + json.dumps(payload, separators=(",", ":")))
'''


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROCKETBOX_ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _authenticated_record(
    relative_path: Path,
    *,
    size_bytes: int,
    sha256: str,
    git_blob_sha1: str,
) -> dict[str, object]:
    path = (ROCKETBOX_ROOT / relative_path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise NativeMaterialCanaryError(f"official input is not a direct file: {path}")
    if path.stat().st_size != size_bytes:
        raise NativeMaterialCanaryError(f"official input size changed: {path}")
    if _sha256_file(path) != sha256:
        raise NativeMaterialCanaryError(f"official input SHA-256 changed: {path}")
    current_blob = _git("hash-object", str(path))
    if current_blob != git_blob_sha1:
        raise NativeMaterialCanaryError(f"official input Git blob changed: {path}")
    tree_line = _git("ls-tree", ROCKETBOX_COMMIT, relative_path.as_posix())
    fields = tree_line.split(None, 3)
    if len(fields) != 4 or fields[:3] != ["100644", "blob", git_blob_sha1]:
        raise NativeMaterialCanaryError(f"official Git tree entry changed: {path}")
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "git_blob_sha1": git_blob_sha1,
    }


def parse_tga_bytes(raw: bytes) -> dict[str, object]:
    if len(raw) < 44:
        raise NativeMaterialCanaryError("TGA is too short")
    header = raw[:18]
    id_length = header[0]
    color_map_type = header[1]
    image_type = header[2]
    width = int.from_bytes(header[12:14], "little")
    height = int.from_bytes(header[14:16], "little")
    pixel_depth = header[16]
    descriptor = header[17]
    if id_length != 0 or color_map_type != 0:
        raise NativeMaterialCanaryError("TGA has an unsupported ID or color map")
    if image_type != 2:
        raise NativeMaterialCanaryError("TGA must be uncompressed true-color type 2")
    if width != 2048 or height != 2048 or pixel_depth != 24 or descriptor != 0:
        raise NativeMaterialCanaryError("TGA dimensions, depth, or origin changed")
    payload_size = width * height * 3
    expected_size = 18 + payload_size + 26
    if len(raw) != expected_size:
        raise NativeMaterialCanaryError("TGA byte size does not match its header")
    footer = raw[-26:]
    if footer != b"\x00" * 8 + b"TRUEVISION-XFILE.\x00":
        raise NativeMaterialCanaryError("TGA footer is not canonical")
    payload = raw[18 : 18 + payload_size]
    bottom_up_bgr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    rgb = bottom_up_bgr[::-1, :, ::-1].copy()
    return {
        "raw_bytes": raw,
        "header_bytes": header,
        "pixel_payload": payload,
        "footer_bytes": footer,
        "image_type": image_type,
        "width": width,
        "height": height,
        "pixel_depth": pixel_depth,
        "descriptor": descriptor,
        "origin": "bottom_left",
        "rgb": rgb,
    }


def load_authenticated_source() -> dict[str, object]:
    try:
        repository_commit = _git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        raise NativeMaterialCanaryError("could not authenticate Rocketbox Git tree") from error
    if repository_commit != ROCKETBOX_COMMIT:
        raise NativeMaterialCanaryError("Rocketbox repository commit changed")
    records = {
        name: _authenticated_record(
            spec["relative_path"],
            size_bytes=spec["size_bytes"],
            sha256=spec["sha256"],
            git_blob_sha1=spec["git_blob_sha1"],
        )
        for name, spec in SOURCE_RECORDS.items()
    }
    protected = {}
    for role, (filename, size_bytes, sha256, git_blob_sha1) in (
        PROTECTED_TEXTURE_SPECS.items()
    ):
        protected[role] = _authenticated_record(
            TEXTURE_ROOT_RELATIVE / filename,
            size_bytes=size_bytes,
            sha256=sha256,
            git_blob_sha1=git_blob_sha1,
        )
    body_color_path = Path(records["body_color"]["path"])
    tga = parse_tga_bytes(body_color_path.read_bytes())
    return {
        "repository_commit": repository_commit,
        "fbx": records["fbx"],
        "body_color": records["body_color"],
        "protected_textures": protected,
        "tga": tga,
    }


def extract_authenticated_shirt_uv(source: dict[str, object]) -> dict[str, object]:
    if source.get("repository_commit") != ROCKETBOX_COMMIT:
        raise NativeMaterialCanaryError("unauthenticated source passed to UV extractor")
    fbx_record = source.get("fbx")
    if not isinstance(fbx_record, dict) or fbx_record.get("sha256") != SOURCE_RECORDS[
        "fbx"
    ]["sha256"]:
        raise NativeMaterialCanaryError("unauthenticated FBX passed to UV extractor")
    blender = BLENDER_PATH.absolute()
    if blender.is_symlink() or not blender.is_file() or not os.access(blender, os.X_OK):
        raise NativeMaterialCanaryError("pinned Blender executable is unavailable")
    completed = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-expr",
            _BLENDER_UV_EXTRACT_CODE,
            "--",
            str(fbx_record["path"]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise NativeMaterialCanaryError(
            f"Blender UV extraction failed with return code {completed.returncode}"
        )
    marker = "ROCKETBOX_NATIVE_UV_JSON="
    payload_lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(payload_lines) != 1:
        raise NativeMaterialCanaryError("Blender UV extraction emitted no unique payload")
    try:
        payload = json.loads(payload_lines[0][len(marker) :])
    except json.JSONDecodeError as error:
        raise NativeMaterialCanaryError("Blender UV payload is not valid JSON") from error
    expected_projection = {
        "mesh_name": MESH_NAME,
        "material_names": ["m002_body", "m002_head", "m002_opacity"],
        "material_polygon_counts": [3939, 3007, 494],
        "body_uv_island_count": 18,
        "shirt_island_signatures": SHIRT_ISLAND_SIGNATURES,
    }
    for key, expected in expected_projection.items():
        if payload.get(key) != expected:
            raise NativeMaterialCanaryError(f"fixed Rocketbox UV contract changed: {key}")
    indices = np.asarray(payload.get("polygon_indices"), dtype="<u4")
    polygons = payload.get("uv_polygons")
    if (
        indices.shape != (SHIRT_FACE_COUNT,)
        or len(np.unique(indices)) != SHIRT_FACE_COUNT
        or not isinstance(polygons, list)
        or len(polygons) != SHIRT_FACE_COUNT
    ):
        raise NativeMaterialCanaryError("fixed Rocketbox shirt face set changed")
    if hashlib.sha256(indices.tobytes()).hexdigest() != SHIRT_FACE_INDICES_U32LE_SHA256:
        raise NativeMaterialCanaryError("fixed Rocketbox shirt face index hash changed")
    for polygon in polygons:
        values = np.asarray(polygon, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] < 3
            or values.shape[1] != 2
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise NativeMaterialCanaryError("fixed Rocketbox shirt UV polygon is invalid")
    return payload


def rasterize_uv_polygons(
    polygons: list[list[list[float]]], *, width: int, height: int
) -> np.ndarray:
    """Rasterize UV polygons at pixel centers with deterministic barycentrics."""
    if width <= 1 or height <= 1:
        raise NativeMaterialCanaryError("UV raster dimensions are invalid")
    mask = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        values = np.asarray(polygon, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] < 3
            or values.shape[1] != 2
            or not np.isfinite(values).all()
        ):
            raise NativeMaterialCanaryError("UV polygon is invalid")
        for offset in range(1, len(values) - 1):
            uv = values[[0, offset, offset + 1]]
            points = np.empty((3, 2), dtype=np.float64)
            points[:, 0] = uv[:, 0] * (width - 1)
            points[:, 1] = (1.0 - uv[:, 1]) * (height - 1)
            x_min = max(0, int(np.floor(points[:, 0].min())))
            x_max = min(width - 1, int(np.ceil(points[:, 0].max())))
            y_min = max(0, int(np.floor(points[:, 1].min())))
            y_max = min(height - 1, int(np.ceil(points[:, 1].max())))
            if x_max < x_min or y_max < y_min:
                continue
            x0, y0 = points[0]
            x1, y1 = points[1]
            x2, y2 = points[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(float(denominator)) <= 1.0e-20:
                continue
            x = np.arange(x_min, x_max + 1, dtype=np.float64)[None, :] + 0.5
            y = np.arange(y_min, y_max + 1, dtype=np.float64)[:, None] + 0.5
            first = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
            second = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
            third = 1.0 - first - second
            inside = (first >= -1.0e-9) & (second >= -1.0e-9) & (third >= -1.0e-9)
            mask[y_min : y_max + 1, x_min : x_max + 1] |= inside
    return mask


def _dilate_eight_connected(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    output = np.zeros_like(values)
    for row_offset in range(3):
        for column_offset in range(3):
            output |= padded[
                row_offset : row_offset + values.shape[0],
                column_offset : column_offset + values.shape[1],
            ]
    return output


def build_shirt_masks(
    source: dict[str, object], uv_bundle: dict[str, object]
) -> dict[str, object]:
    tga = source.get("tga")
    if not isinstance(tga, dict) or tga.get("width") != 2048 or tga.get("height") != 2048:
        raise NativeMaterialCanaryError("unauthenticated TGA passed to mask builder")
    if uv_bundle.get("shirt_island_signatures") != SHIRT_ISLAND_SIGNATURES:
        raise NativeMaterialCanaryError("unauthenticated UV bundle passed to mask builder")
    height = int(tga["height"])
    width = int(tga["width"])
    surface = rasterize_uv_polygons(
        uv_bundle["uv_polygons"], width=width, height=height
    )
    if int(surface.sum()) != 1_129_206:
        raise NativeMaterialCanaryError("fixed Rocketbox shirt surface mask changed")

    rgb = np.asarray(tga["rgb"], dtype=np.uint32)
    luma8 = (
        54 * rgb[:, :, 0] + 183 * rgb[:, :, 1] + 19 * rgb[:, :, 2] + 128
    ) >> 8
    stripe_core = surface & (luma8 < STRIPE_LUMA_THRESHOLD)
    protected = _dilate_eight_connected(stripe_core)
    y_grid, x_grid = np.ogrid[:height, :width]
    for x, y, radius in BUTTON_GUARDS:
        protected |= (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
    protected &= surface
    main = surface & ~protected
    if not np.array_equal(surface, protected | main) or np.any(protected & main):
        raise NativeMaterialCanaryError("shirt semantic masks do not partition the surface")
    masks = {
        "shirt_surface": surface,
        "stripe_detail_protect": protected,
        "shirt_main_color": main,
    }
    return {
        **masks,
        "mask_sha256": {
            name: hashlib.sha256(mask.tobytes()).hexdigest()
            for name, mask in masks.items()
        },
        "pixel_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "construction": {
            "uv_origin": "bottom_left_to_top_left_y_flip",
            "polygon_fill": "pixel_center_barycentric_tolerance_1e-9",
            "stripe_luma": "integer_rec709_y8_lt_145",
            "stripe_boundary_dilation_pixels": 1,
            "button_guards": [list(value) for value in BUTTON_GUARDS],
        },
    }


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64) / 255.0
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * clipped ** (1.0 / 2.4) - 0.055,
    )
    return np.rint(srgb * 255.0).astype(np.uint8)


def build_tga_variant(
    source: dict[str, object],
    masks: dict[str, object],
    palette_srgb: list[int] | tuple[int, int, int],
) -> dict[str, object]:
    """Apply one registered sRGB color while preserving all non-target bytes."""
    tga = source.get("tga")
    if not isinstance(tga, dict) or _sha256_bytes(tga.get("raw_bytes", b"")) != (
        SOURCE_BODY_COLOR_SHA256
    ):
        raise NativeMaterialCanaryError("unauthenticated TGA passed to variant builder")
    shape = (int(tga["height"]), int(tga["width"]))
    surface = np.asarray(masks.get("shirt_surface"), dtype=bool)
    protected = np.asarray(masks.get("stripe_detail_protect"), dtype=bool)
    main = np.asarray(masks.get("shirt_main_color"), dtype=bool)
    if (
        surface.shape != shape
        or protected.shape != shape
        or main.shape != shape
        or not np.array_equal(surface, protected | main)
        or np.any(protected & main)
        or not main.any()
    ):
        raise NativeMaterialCanaryError("invalid shirt masks passed to variant builder")

    original_rgb = np.asarray(tga["rgb"], dtype=np.uint8)
    output_rgb = original_rgb.copy()
    source_linear = _srgb_to_linear(original_rgb[main])
    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    source_luminance = source_linear @ luminance_weights
    reference_luminance = float(np.median(source_luminance))
    if not np.isfinite(reference_luminance) or reference_luminance <= 1.0e-12:
        raise NativeMaterialCanaryError("shirt main-color mask has unusable luminance")
    if (
        not isinstance(palette_srgb, (list, tuple))
        or len(palette_srgb) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or channel < 0
            or channel > 255
            for channel in palette_srgb
        )
    ):
        raise NativeMaterialCanaryError("palette_srgb must contain three uint8 values")
    palette = np.asarray(palette_srgb, dtype=np.uint8)
    palette_linear = _srgb_to_linear(palette.reshape(1, 3))[0]
    scale = np.clip(source_luminance / reference_luminance, 0.30, 2.25)
    output_rgb[main] = _linear_to_srgb(palette_linear[None, :] * scale[:, None])

    raw = bytearray(tga["raw_bytes"])
    raw_values = np.frombuffer(raw, dtype=np.uint8)
    rows, columns = np.nonzero(main)
    raw_rows = shape[0] - 1 - rows
    offsets = 18 + (raw_rows * shape[1] + columns) * 3
    target = output_rgb[rows, columns]
    raw_values[offsets] = target[:, 2]
    raw_values[offsets + 1] = target[:, 1]
    raw_values[offsets + 2] = target[:, 0]
    output_bytes = bytes(raw)
    decoded = parse_tga_bytes(output_bytes)
    decoded_rgb = np.asarray(decoded["rgb"], dtype=np.uint8)
    if not np.array_equal(decoded_rgb, output_rgb):
        raise NativeMaterialCanaryError("raw BGR patch did not roundtrip exactly")
    changed = np.any(decoded_rgb != original_rgb, axis=2)
    output_luminance = _srgb_to_linear(decoded_rgb[main]) @ luminance_weights
    if np.std(source_luminance) <= 1.0e-12 or np.std(output_luminance) <= 1.0e-12:
        luminance_correlation = 1.0
    else:
        luminance_correlation = float(
            np.corrcoef(source_luminance, output_luminance)[0, 1]
        )
    source_raw = tga["raw_bytes"]
    qa = {
        "transform": "registered_srgb_linear_luminance_scale_v1",
        "palette_srgb": palette.tolist(),
        "main_mask_texels": int(main.sum()),
        "inside_mask_changed_pixels": int(np.count_nonzero(changed & main)),
        "outside_mask_changed_pixels": int(np.count_nonzero(changed & ~main)),
        "protected_changed_pixels": int(np.count_nonzero(changed & protected)),
        "header_changed_bytes": int(
            np.count_nonzero(
                np.frombuffer(output_bytes[:18], dtype=np.uint8)
                != np.frombuffer(source_raw[:18], dtype=np.uint8)
            )
        ),
        "footer_changed_bytes": int(
            np.count_nonzero(
                np.frombuffer(output_bytes[-26:], dtype=np.uint8)
                != np.frombuffer(source_raw[-26:], dtype=np.uint8)
            )
        ),
        "size_unchanged": len(output_bytes) == len(source_raw),
        "source_linear_luminance_median": reference_luminance,
        "linear_luminance_correlation": luminance_correlation,
    }
    if (
        qa["inside_mask_changed_pixels"] != int(main.sum())
        or qa["outside_mask_changed_pixels"] != 0
        or qa["protected_changed_pixels"] != 0
        or qa["header_changed_bytes"] != 0
        or qa["footer_changed_bytes"] != 0
        or not qa["size_unchanged"]
        or luminance_correlation < 0.98
    ):
        raise NativeMaterialCanaryError("native TGA variant failed pixel protection QA")
    return {
        "tga_bytes": output_bytes,
        "rgb": decoded_rgb,
        "source_sha256": _sha256_bytes(source_raw),
        "output_sha256": _sha256_bytes(output_bytes),
        "qa": qa,
    }


def build_blue_tga_variant(
    source: dict[str, object], masks: dict[str, object]
) -> dict[str, object]:
    """Retain the frozen historical blue canary as a generic-transform wrapper."""

    variant = build_tga_variant(source, masks, [36, 88, 207])
    if variant["output_sha256"] != FROZEN_BLUE_TGA_SHA256:
        raise NativeMaterialCanaryError("blue variant hash changed")
    return variant


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.absolute())
        ),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    values = np.asarray(mask, dtype=bool)
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(
        path, format="PNG", optimize=False, compress_level=9
    )


def _build_mask_overlay(source_rgb: np.ndarray, masks: dict[str, object]) -> np.ndarray:
    rgb = np.asarray(source_rgb, dtype=np.uint8)
    surface = np.asarray(masks["shirt_surface"], dtype=bool)
    protected = np.asarray(masks["stripe_detail_protect"], dtype=bool)
    main = np.asarray(masks["shirt_main_color"], dtype=bool)
    overlay = rgb.astype(np.float64)
    overlay[main] = overlay[main] * 0.55 + np.asarray([35, 115, 255]) * 0.45
    overlay[protected] = (
        overlay[protected] * 0.45 + np.asarray([255, 180, 20]) * 0.55
    )
    boundary = surface & _dilate_eight_connected(~surface)
    overlay[boundary] = np.asarray([255, 255, 255])
    return np.rint(np.clip(overlay, 0.0, 255.0)).astype(np.uint8)


def _build_texture_diff(
    source_rgb: np.ndarray, variant_rgb: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.uint8)
    variant = np.asarray(variant_rgb, dtype=np.uint8)
    if source.shape != variant.shape or source.ndim != 3 or source.shape[2] != 3:
        raise NativeMaterialCanaryError("texture diff inputs do not match")
    absolute = np.abs(variant.astype(np.int16) - source.astype(np.int16))
    amplified = np.clip(absolute * 4, 0, 255).astype(np.uint8)
    height = 1024
    width = 1024
    panels = []
    for values in (source, variant, amplified):
        panels.append(
            np.asarray(
                Image.fromarray(values, mode="RGB").resize(
                    (width, height), resample=Image.Resampling.LANCZOS
                )
            )
        )
    separator = np.full((height, 8, 3), 24, dtype=np.uint8)
    return np.concatenate(
        [panels[0], separator, panels[1], separator, panels[2]], axis=1
    )


def _seal_readonly_tree(root: Path) -> None:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=str)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in files:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    for path in directories:
        path.chmod(0o555)
    root.chmod(0o555)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _remove_staging_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_dir():
            child.chmod(0o755)
        else:
            child.chmod(0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def write_canary_output(
    output: Path,
    source: dict[str, object],
    uv_bundle: dict[str, object],
    masks: dict[str, object],
    variant: dict[str, object],
) -> Path:
    """Publish immutable, reference-only material canary evidence once."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise NativeMaterialCanaryError(f"refusing to replace existing output: {output}")
    if source.get("repository_commit") != ROCKETBOX_COMMIT:
        raise NativeMaterialCanaryError("unauthenticated source passed to writer")
    if variant.get("source_sha256") != SOURCE_BODY_COLOR_SHA256:
        raise NativeMaterialCanaryError("variant source hash changed")
    if variant.get("output_sha256") != FROZEN_BLUE_TGA_SHA256:
        raise NativeMaterialCanaryError("blue variant hash changed")
    if uv_bundle.get("shirt_island_signatures") != SHIRT_ISLAND_SIGNATURES:
        raise NativeMaterialCanaryError("shirt UV contract changed before publication")
    polygon_indices = np.asarray(uv_bundle.get("polygon_indices"), dtype="<u4")
    if (
        polygon_indices.shape != (SHIRT_FACE_COUNT,)
        or _sha256_bytes(polygon_indices.tobytes())
        != SHIRT_FACE_INDICES_U32LE_SHA256
    ):
        raise NativeMaterialCanaryError("shirt polygon registry changed")
    if masks.get("pixel_counts") != {
        name: record["pixel_count"] for name, record in FROZEN_MASK_REGISTRY.items()
    } or masks.get("mask_sha256") != {
        name: record["raw_bool_sha256"]
        for name, record in FROZEN_MASK_REGISTRY.items()
    }:
        raise NativeMaterialCanaryError("frozen shirt mask registry changed")
    qa = variant.get("qa")
    if not isinstance(qa, dict) or any(
        (
            qa.get("outside_mask_changed_pixels") != 0,
            qa.get("protected_changed_pixels") != 0,
            qa.get("header_changed_bytes") != 0,
            qa.get("footer_changed_bytes") != 0,
            qa.get("size_unchanged") is not True,
            float(qa.get("linear_luminance_correlation", 0.0)) < 0.98,
        )
    ):
        raise NativeMaterialCanaryError("variant QA is not publishable")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise NativeMaterialCanaryError(f"unexpected staging collision: {staging}")
    try:
        masks_dir = staging / "masks"
        diagnostics_dir = staging / "diagnostics"
        variant_dir = staging / "variant"
        for directory in (masks_dir, diagnostics_dir, variant_dir):
            directory.mkdir(parents=True, exist_ok=False)

        for name in FROZEN_MASK_REGISTRY:
            _save_mask_png(masks_dir / f"{name}.png", masks[name])
        with (masks_dir / "shirt_polygon_indices.npy").open("wb") as stream:
            np.save(stream, polygon_indices, allow_pickle=False)
        Image.fromarray(
            _build_mask_overlay(source["tga"]["rgb"], masks), mode="RGB"
        ).save(
            diagnostics_dir / "mask_overlay.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        Image.fromarray(
            _build_texture_diff(source["tga"]["rgb"], variant["rgb"]),
            mode="RGB",
        ).save(
            diagnostics_dir / "texture_diff.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        variant_path = variant_dir / "m002_body_color.tga"
        variant_path.write_bytes(variant["tga_bytes"])

        mask_registry = {
            "schema": "rocketbox_native_semantic_mask_registry_v1",
            "asset_id": ASSET_ID,
            "source_body_color_sha256": SOURCE_BODY_COLOR_SHA256,
            "face_count": SHIRT_FACE_COUNT,
            "face_indices_dtype": "uint32_little_endian",
            "face_indices_u32le_sha256": SHIRT_FACE_INDICES_U32LE_SHA256,
            "shirt_island_signatures": SHIRT_ISLAND_SIGNATURES,
            "masks": FROZEN_MASK_REGISTRY,
            "construction": masks["construction"],
        }
        mask_registry_path = staging / "mask_registry.json"
        _write_json(mask_registry_path, mask_registry)

        artifact_paths = sorted(
            (path for path in staging.rglob("*") if path.is_file()), key=str
        )
        protected = source.get("protected_textures")
        if not isinstance(protected, dict) or set(protected) != set(
            PROTECTED_TEXTURE_SPECS
        ):
            raise NativeMaterialCanaryError("protected texture references changed")
        manifest = {
            "schema": "rocketbox_native_material_canary_v1",
            "asset_id": ASSET_ID,
            "variant_id": "shirt_blue_v1",
            "tag": f"{ASSET_ID}_shirt_blue_v1",
            "usage_scope": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "source": {
                "repository_root": str(ROCKETBOX_ROOT),
                "repository_commit": ROCKETBOX_COMMIT,
                "fbx": source["fbx"],
                "body_color": source["body_color"],
                "protected_texture_references": protected,
            },
            "mask_registry": {
                "path": "mask_registry.json",
                "sha256": _sha256_file(mask_registry_path),
                "face_count": SHIRT_FACE_COUNT,
                "face_indices_u32le_sha256": SHIRT_FACE_INDICES_U32LE_SHA256,
                "masks": FROZEN_MASK_REGISTRY,
            },
            "variant": {
                "path": "variant/m002_body_color.tga",
                "target_image_name": "m002_body_color",
                "source_sha256": SOURCE_BODY_COLOR_SHA256,
                "sha256": variant["output_sha256"],
                "size_bytes": len(variant["tga_bytes"]),
                "qa": qa,
            },
            "diagnostics": {
                "mask_overlay": "diagnostics/mask_overlay.png",
                "texture_diff": "diagnostics/texture_diff.png",
                "texture_diff_panels": [
                    "source_body_color",
                    "blue_variant",
                    "absolute_rgb_difference_x4",
                ],
            },
            "artifacts": {
                path.relative_to(staging).as_posix(): _file_record(
                    path, relative_to=staging
                )
                for path in artifact_paths
            },
            "automatic_qa": {
                "all_checks_passed": True,
                "source_hash_locked": True,
                "shirt_uv_faces_hash_locked": True,
                "semantic_masks_hash_locked": True,
                "non_target_pixels_unchanged": True,
                "protected_details_unchanged": True,
                "tga_container_unchanged": True,
                "variant_hash_locked": True,
            },
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        expected = {
            "manifest.json",
            "mask_registry.json",
            "masks/shirt_surface.png",
            "masks/stripe_detail_protect.png",
            "masks/shirt_main_color.png",
            "masks/shirt_polygon_indices.npy",
            "diagnostics/mask_overlay.png",
            "diagnostics/texture_diff.png",
            "variant/m002_body_color.tga",
        }
        actual = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise NativeMaterialCanaryError("canary output inventory changed")
        _seal_readonly_tree(staging)
        if output.exists() or output.is_symlink():
            raise NativeMaterialCanaryError(
                f"refusing to replace concurrently-created output: {output}"
            )
        os.rename(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return output / "manifest.json"
    except Exception:
        _remove_staging_tree(staging)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = load_authenticated_source()
    uv_bundle = extract_authenticated_shirt_uv(source)
    masks = build_shirt_masks(source, uv_bundle)
    variant = build_blue_tga_variant(source, masks)
    manifest_path = write_canary_output(
        args.output, source, uv_bundle, masks, variant
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": str(manifest_path),
                "variant_sha256": variant["output_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
