"""Losslessly transcode embedded EXT_texture_webp images to core glTF PNG.

Geometry, skinning, inverse binds, animations, and all existing buffer bytes
are left untouched.  New PNG payloads are appended and texture references are
rewritten for importers such as UE 5.5 that reject required EXT_texture_webp.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
from pathlib import Path

from PIL import Image


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_glb(path: Path):
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError("input is not GLB 2.0")
    version, declared = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared != len(raw):
        raise ValueError("invalid GLB header")
    offset = 12
    chunks = {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[kind] = raw[offset : offset + length]
        offset += length
    if set(chunks) != {JSON_CHUNK, BIN_CHUNK}:
        raise ValueError("GLB must contain one JSON and one BIN chunk")
    document = json.loads(chunks[JSON_CHUNK].decode("utf-8").rstrip(" \x00"))
    binary_length = int(document["buffers"][0]["byteLength"])
    return document, chunks[BIN_CHUNK][:binary_length]


def encode_glb(document, binary: bytes) -> bytes:
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary_padded = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary_padded)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(json_bytes), JSON_CHUNK),
            json_bytes,
            struct.pack("<II", len(binary_padded), BIN_CHUNK),
            binary_padded,
        )
    )


def _append_aligned(binary: bytearray, payload: bytes) -> int:
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    offset = len(binary)
    binary.extend(payload)
    return offset


def _write_all_exclusive(path: Path, payload: bytes) -> None:
    """Publish one complete file without replacing an existing artifact."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def transcode(document: dict, binary: bytes):
    images = document.get("images", [])
    textures = document.get("textures", [])
    buffer_views = document.get("bufferViews", [])
    if not images or not textures or not buffer_views:
        raise ValueError("GLB has no embedded texture graph")
    result_binary = bytearray(binary)
    records = []
    converted_indices = set()
    for index, image in enumerate(images):
        if image.get("mimeType") != "image/webp":
            continue
        view_index = image.get("bufferView")
        if not isinstance(view_index, int) or not 0 <= view_index < len(buffer_views):
            raise ValueError(f"image {index} has an invalid bufferView")
        view = buffer_views[view_index]
        if view.get("buffer", 0) != 0:
            raise ValueError("only the embedded buffer is supported")
        start = int(view.get("byteOffset", 0))
        end = start + int(view["byteLength"])
        source = binary[start:end]
        with Image.open(io.BytesIO(source)) as opened:
            opened.load()
            rgba = opened.convert("RGBA")
            pixels = rgba.tobytes()
            size = list(rgba.size)
            encoded = io.BytesIO()
            rgba.save(encoded, format="PNG", optimize=False, compress_level=6)
            png = encoded.getvalue()
        offset = _append_aligned(result_binary, png)
        new_view = {"buffer": 0, "byteOffset": offset, "byteLength": len(png)}
        buffer_views.append(new_view)
        image["bufferView"] = len(buffer_views) - 1
        image["mimeType"] = "image/png"
        converted_indices.add(index)
        records.append(
            {
                "image_index": index,
                "pixel_size": size,
                "rgba_sha256": hashlib.sha256(pixels).hexdigest(),
                "source_webp_sha256": hashlib.sha256(source).hexdigest(),
                "source_size_bytes": len(source),
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "png_size_bytes": len(png),
            }
        )
    if not converted_indices:
        raise ValueError("GLB contains no WebP images to transcode")
    for texture in textures:
        extensions = texture.get("extensions")
        webp = extensions.get("EXT_texture_webp") if isinstance(extensions, dict) else None
        if isinstance(webp, dict):
            source = webp.get("source")
            if source not in converted_indices:
                raise ValueError("texture WebP source was not converted")
            texture["source"] = source
            del extensions["EXT_texture_webp"]
            if not extensions:
                texture.pop("extensions", None)
    for key in ("extensionsUsed", "extensionsRequired"):
        values = [value for value in document.get(key, []) if value != "EXT_texture_webp"]
        if values:
            document[key] = values
        else:
            document.pop(key, None)
    return document, bytes(result_binary), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("output or manifest already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document, binary = read_glb(input_path)
    before_structure = {
        key: document.get(key)
        for key in ("meshes", "skins", "nodes", "accessors", "animations", "scenes", "scene")
    }
    rewritten, rewritten_binary, records = transcode(document, binary)
    payload = encode_glb(rewritten, rewritten_binary)
    _write_all_exclusive(output_path, payload)
    readback, _ = read_glb(output_path)
    after_structure = {
        key: readback.get(key)
        for key in ("meshes", "skins", "nodes", "accessors", "animations", "scenes", "scene")
    }
    if before_structure != after_structure:
        raise RuntimeError("transcode changed geometry, skin, hierarchy, or animation")
    if any(image.get("mimeType") != "image/png" for image in readback.get("images", [])):
        raise RuntimeError("PNG readback failed")
    if "EXT_texture_webp" in readback.get("extensionsRequired", []):
        raise RuntimeError("WebP remains required after transcode")
    manifest = {
        "schema": "glb_embedded_webp_to_png_transcode_v1",
        "purpose": "UE_5.5_interchange_compatibility",
        "geometry_skin_animation_byte_graph_changed": False,
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "size_bytes": input_path.stat().st_size,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "images": records,
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_all_exclusive(manifest_path, encoded_manifest)
    print(f"GLB_WEBP_TO_PNG_OK output={output_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
