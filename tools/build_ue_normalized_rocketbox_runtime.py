#!/usr/bin/env python3
"""Publish immutable grounded/metric native Rocketbox runtimes for Unreal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.normalize_rocketbox_glb_for_ue import (
    normalize_grounded_metric_glb_bytes,
    read_glb_bytes,
)


CONTRACTS = {
    "rocketbox_male_adult_01_original_ue_v2": {
        "asset_id": "rocketbox_male_adult_01",
        "variant_id": "original_v1",
        "source_tag": "rocketbox_male_adult_01_original_v1",
        "source_relative_root": (
            "tmp/rocketbox_native_runtime_v1/"
            "rocketbox_male_adult_01_original_v1"
        ),
        "source_manifest": "build_manifest.json",
        "source_manifest_schema": "rocketbox_native_runtime_build_v1",
        "output_relative_root": (
            "tmp/rocketbox_native_runtime_ue_v2/"
            "rocketbox_male_adult_01_original_ue_v2"
        ),
    },
    "rocketbox_male_adult_01_shirt_blue_ue_v2": {
        "asset_id": "rocketbox_male_adult_01",
        "variant_id": "shirt_blue_v1",
        "source_tag": "rocketbox_male_adult_01_shirt_blue_v1",
        "source_relative_root": (
            "tmp/rocketbox_native_runtime_v1/"
            "rocketbox_male_adult_01_shirt_blue_v1"
        ),
        "source_manifest": "variant_manifest.json",
        "source_manifest_schema": "rocketbox_native_material_variant_v1",
        "output_relative_root": (
            "tmp/rocketbox_native_runtime_ue_v2/"
            "rocketbox_male_adult_01_shirt_blue_ue_v2"
        ),
    },
}


class UeRuntimeBundleError(RuntimeError):
    """Raised when a source or normalized runtime fails its immutable contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(tag: str, contract: dict[str, object]):
    source_root = (SPEAR_ROOT / contract["source_relative_root"]).resolve()
    source_glb = (source_root / "runtime.glb").resolve()
    source_manifest_path = (source_root / contract["source_manifest"]).resolve()
    for path in (source_glb, source_manifest_path):
        if path.is_symlink() or not path.is_file() or path.parent != source_root:
            raise UeRuntimeBundleError(f"source is not a direct regular file: {path}")
    try:
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UeRuntimeBundleError("source manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != contract["source_manifest_schema"]
        or manifest.get("tag") != contract["source_tag"]
        or manifest.get("asset_id") != contract["asset_id"]
        or manifest.get("variant_id") != contract["variant_id"]
        or manifest.get("usage_scope") != "research_candidate"
        or manifest.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise UeRuntimeBundleError(f"source manifest contract changed for {tag}")
    runtime = manifest.get("runtime_glb")
    if (
        not isinstance(runtime, dict)
        or runtime.get("filename") != "runtime.glb"
        or runtime.get("size_bytes") != source_glb.stat().st_size
        or runtime.get("sha256") != _sha256_file(source_glb)
    ):
        raise UeRuntimeBundleError("source runtime no longer matches its manifest")
    return source_root, source_glb, source_manifest_path, manifest


def _buffer_view_bytes(document, binary, index):
    views = document.get("bufferViews", [])
    if not isinstance(index, int) or index < 0 or index >= len(views):
        raise UeRuntimeBundleError("embedded image bufferView is invalid")
    view = views[index]
    if not isinstance(view, dict) or view.get("buffer", 0) != 0:
        raise UeRuntimeBundleError("embedded image escaped the GLB buffer")
    start = int(view.get("byteOffset", 0))
    end = start + int(view.get("byteLength", 0))
    if start < 0 or end > len(binary):
        raise UeRuntimeBundleError("embedded image bufferView exceeds the GLB")
    return binary[start:end]


def _image_payloads(document, binary):
    records = {}
    for image in document.get("images", []):
        if (
            not isinstance(image, dict)
            or not isinstance(image.get("name"), str)
            or "bufferView" not in image
            or "uri" in image
        ):
            raise UeRuntimeBundleError("runtime image is not uniquely embedded")
        payload = _buffer_view_bytes(document, binary, image["bufferView"])
        records[image["name"]] = {
            "mime_type": image.get("mimeType"),
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    if len(records) != 7:
        raise UeRuntimeBundleError("native runtime must retain seven embedded images")
    return records


def _seal_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        path.chmod(0o555)
    root.chmod(0o555)


def _cleanup(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def publish_bundle(
    tag: str, contract: dict[str, object], output: Path
) -> Path:
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise UeRuntimeBundleError(f"refusing to replace UE runtime bundle: {output}")
    source_root, source_glb, source_manifest_path, source_manifest = _load_source(
        tag, contract
    )
    source_payload = source_glb.read_bytes()
    normalized_payload, normalization = normalize_grounded_metric_glb_bytes(
        source_payload
    )
    source_document, source_binary = read_glb_bytes(source_payload)
    normalized_document, normalized_binary = read_glb_bytes(normalized_payload)
    source_images = _image_payloads(source_document, source_binary)
    normalized_images = _image_payloads(normalized_document, normalized_binary)
    material_graph_keys = ("materials", "textures", "samplers", "images")
    material_graph_unchanged = all(
        source_document.get(key, []) == normalized_document.get(key, [])
        for key in material_graph_keys
    )
    source_animation_names = [
        animation.get("name") for animation in source_document.get("animations", [])
    ]
    normalized_animation_names = [
        animation.get("name")
        for animation in normalized_document.get("animations", [])
    ]
    if (
        source_images != normalized_images
        or not material_graph_unchanged
        or source_animation_names != normalized_animation_names
        or set(normalized_animation_names) != {"Walking", "Standing_Idle"}
        or normalization.get("normalized_joint_count") != 80
        or normalization.get("static_wrapper_translation_zeroed") is not True
    ):
        raise UeRuntimeBundleError("normalized runtime equivalence checks failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        runtime_path = staging / "runtime.glb"
        runtime_path.write_bytes(normalized_payload)
        if runtime_path.read_bytes() != normalized_payload:
            raise UeRuntimeBundleError("normalized runtime readback changed")
        runtime_record = {
            "filename": "runtime.glb",
            "size_bytes": runtime_path.stat().st_size,
            "sha256": _sha256_file(runtime_path),
        }
        manifest = {
            "schema": "rocketbox_native_ue_runtime_v2",
            "tag": tag,
            "source_tag": contract["source_tag"],
            "asset_id": contract["asset_id"],
            "variant_id": contract["variant_id"],
            "usage_scope": "research_candidate",
            "formal_dataset_asset": False,
            "formal_registration_authorized": False,
            "runtime_glb": runtime_record,
            "source": {
                "root": str(source_root),
                "runtime_glb": str(source_glb),
                "runtime_glb_sha256": _sha256_file(source_glb),
                "runtime_glb_size_bytes": source_glb.stat().st_size,
                "manifest": str(source_manifest_path),
                "manifest_sha256": _sha256_file(source_manifest_path),
                "manifest_schema": source_manifest["schema"],
            },
            "license": source_manifest.get("license"),
            "normalization": normalization,
            "equivalence": {
                "mesh_position_accessors_unchanged": True,
                "embedded_image_payloads_unchanged": source_images
                == normalized_images,
                "material_texture_graph_unchanged": material_graph_unchanged,
                "animation_names_unchanged": source_animation_names
                == normalized_animation_names,
                "embedded_image_payloads": normalized_images,
            },
            "expected_ue_qa": {
                "demographic": "adult",
                "height_range_cm": [165.0, 200.0],
                "bottom_range_cm": [-5.0, 5.0],
                "actor_scale": 1.0,
            },
            "automatic_checks": {
                "overall": "passed",
                "source_hash_locked": "passed",
                "grounded_metric_normalization": "passed",
                "mesh_positions": "unchanged",
                "embedded_images": "unchanged",
                "material_texture_graph": "unchanged",
                "animation_names": "unchanged",
            },
        }
        manifest_path = staging / "normalization_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if {path.name for path in staging.iterdir()} != {
            "runtime.glb",
            "normalization_manifest.json",
        }:
            raise UeRuntimeBundleError("normalized runtime inventory changed")
        _seal_tree(staging)
        if output.exists() or output.is_symlink():
            raise UeRuntimeBundleError(
                f"refusing to replace concurrently-created bundle: {output}"
            )
        os.rename(staging, output)
        return output / "normalization_manifest.json"
    except Exception:
        _cleanup(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=sorted(CONTRACTS), required=True)
    arguments = parser.parse_args()
    contract = CONTRACTS[arguments.tag]
    output = SPEAR_ROOT / contract["output_relative_root"]
    manifest = publish_bundle(arguments.tag, contract, output)
    print(
        json.dumps(
            {
                "status": "passed",
                "tag": arguments.tag,
                "manifest": str(manifest),
                "manifest_sha256": _sha256_file(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
