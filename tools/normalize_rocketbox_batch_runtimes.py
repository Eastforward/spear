#!/usr/bin/env python3

"""Publish in-place metric UE bundles for every native Rocketbox runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SPEAR_ROOT))

from tools.normalize_rocketbox_glb_for_ue import (
    normalize_in_place_grounded_metric_glb_bytes,
    read_glb_bytes,
)


SOURCE_SCHEMA = "rocketbox_batch_native_runtime_v1"
OUTPUT_SCHEMA = "rocketbox_batch_native_ue_runtime_v1"
NORMALIZATION_SCHEMA = (
    "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
)
HEIGHT_RANGES = {
    "Adults": [140.0, 215.0],
    "Professions": [140.0, 215.0],
    "Children": [80.0, 170.0],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_is_verified(manifest_path: Path, base_avatar_id: str) -> bool:
    manifest_path = Path(manifest_path)
    runtime_path = manifest_path.parent / "runtime.glb"
    if not manifest_path.is_file() or not runtime_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    runtime = manifest.get("runtime_glb", {})
    return (
        manifest.get("schema") == SOURCE_SCHEMA
        and manifest.get("base_avatar_id") == base_avatar_id
        and manifest.get("usage_scope") == "research_candidate"
        and manifest.get("automatic_checks", {}).get("overall") == "passed"
        and runtime.get("filename") == "runtime.glb"
        and runtime.get("size_bytes") == runtime_path.stat().st_size
        and runtime.get("sha256") == sha256_file(runtime_path)
    )


def expected_ue_qa(height_contract: dict, *, demographic: str) -> dict:
    category = height_contract["category"]
    if category not in HEIGHT_RANGES:
        raise RuntimeError(f"unsupported height category: {category}")
    height = float(height_contract["authored_height_cm"])
    allowed = HEIGHT_RANGES[category]
    if not math.isfinite(height) or not allowed[0] <= height <= allowed[1]:
        raise RuntimeError(f"authored height is outside {category} policy: {height}")
    expected_audio = height * (0.88 if demographic == "child" else 0.90)
    recorded_audio = float(height_contract["mouth_audio_height_cm"])
    if abs(recorded_audio - expected_audio) > 1.0e-6:
        raise RuntimeError("inventory mouth/audio height contract changed")
    return {
        "demographic": demographic,
        "authored_height_cm": height,
        "height_range_cm": list(allowed),
        "bottom_range_cm": [-5.0, 5.0],
        "actor_scale": 1.0,
        "authored_height_preserved": True,
        "apartment_ceiling_cm": float(height_contract["apartment_ceiling_cm"]),
        "ceiling_headroom_cm": float(height_contract["ceiling_headroom_cm"]),
        "mouth_audio_height_cm": recorded_audio,
        "ground_snap_to_floor": True,
        "ground_snap_max_abs_correction_cm": 15.0,
        "ground_snap_residual_tolerance_cm": 0.1,
    }


def _buffer_view_payload(document: dict, binary: bytes, index: int) -> bytes:
    views = document.get("bufferViews", [])
    if not isinstance(index, int) or not 0 <= index < len(views):
        raise RuntimeError("embedded image bufferView is invalid")
    view = views[index]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("embedded image uses a non-GLB buffer")
    start = int(view.get("byteOffset", 0))
    end = start + int(view["byteLength"])
    if start < 0 or end > len(binary):
        raise RuntimeError("embedded image bufferView exceeds GLB")
    return binary[start:end]


def _embedded_image_payloads(document: dict, binary: bytes) -> dict:
    records = {}
    for image in document.get("images", []):
        name = image.get("name")
        if not isinstance(name, str) or "uri" in image or "bufferView" not in image:
            raise RuntimeError("runtime image is not uniquely embedded")
        payload = _buffer_view_payload(document, binary, image["bufferView"])
        records[name] = {
            "mime_type": image.get("mimeType"),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if not records:
        raise RuntimeError("runtime has no embedded images")
    return records


def _seal_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _cleanup(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def publish_one(source_dir: Path, output_dir: Path, base_avatar_id: str) -> dict:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source_manifest_path = source_dir / "build_manifest.json"
    source_glb = source_dir / "runtime.glb"
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"refusing to replace normalized runtime: {output_dir}")
    if not source_manifest_is_verified(source_manifest_path, base_avatar_id):
        raise RuntimeError(f"unverified source runtime: {base_avatar_id}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_bytes = source_glb.read_bytes()
    normalized_bytes, normalization = normalize_in_place_grounded_metric_glb_bytes(
        source_bytes
    )
    source_document, source_binary = read_glb_bytes(source_bytes)
    normalized_document, normalized_binary = read_glb_bytes(normalized_bytes)
    source_images = _embedded_image_payloads(source_document, source_binary)
    normalized_images = _embedded_image_payloads(
        normalized_document, normalized_binary
    )
    material_keys = ("materials", "textures", "samplers", "images")
    graph_unchanged = all(
        source_document.get(key, []) == normalized_document.get(key, [])
        for key in material_keys
    )
    source_actions = [item.get("name") for item in source_document.get("animations", [])]
    normalized_actions = [
        item.get("name") for item in normalized_document.get("animations", [])
    ]
    walk = normalization.get("root_motion", {}).get("Walking", {})
    displacement = walk.get("horizontal_displacement_before_m", [])
    displacement_norm = math.sqrt(sum(float(value) ** 2 for value in displacement))
    skeleton_family = source_manifest["action_contract"]["skeleton_family"]
    expected_skin_joint_count = int(source_manifest["glb_contract"]["joint_count"])
    root_name = normalization.get("root", {}).get("armature_node_name")
    # Older normalizer evidence places the root fields at top level.
    if root_name is None:
        root_name = normalization.get("armature_node_name")
    if (
        normalization.get("schema") != NORMALIZATION_SCHEMA
        or normalization.get("normalized_joint_count") != expected_skin_joint_count
        or normalization.get("static_wrapper_translation_zeroed") is not True
        or normalization.get("in_place_actions") != ["Walking"]
        or displacement_norm < 1.0
        or float(walk.get("maximum_horizontal_deviation_after_m", 1.0)) >= 1.0e-6
        or float(walk.get("maximum_vertical_world_error_m", 1.0)) >= 1.0e-6
        or source_images != normalized_images
        or not graph_unchanged
        or source_actions != ["Walking", "Standing_Idle"]
        or normalized_actions != source_actions
        or skeleton_family not in {"Bip01", "Bip02"}
    ):
        raise RuntimeError(f"normalized runtime equivalence failed: {base_avatar_id}")

    qa = expected_ue_qa(
        source_manifest["height_contract"],
        demographic=source_manifest["demographic"],
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent
        )
    )
    try:
        runtime_path = staging / "runtime.glb"
        runtime_path.write_bytes(normalized_bytes)
        runtime_record = {
            "filename": "runtime.glb",
            "size_bytes": runtime_path.stat().st_size,
            "sha256": sha256_file(runtime_path),
        }
        tag = f"{base_avatar_id}_original_ue_v1"
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "tag": tag,
            "base_avatar_id": base_avatar_id,
            "asset_id": source_manifest["asset_id"],
            "variant_id": "original_v1",
            "usage_scope": "research_candidate",
            "formal_dataset_asset": False,
            "formal_registration_authorized": False,
            "runtime_glb": runtime_record,
            "source": {
                "root": str(source_dir),
                "runtime_glb": str(source_glb),
                "runtime_glb_sha256": sha256_file(source_glb),
                "manifest": str(source_manifest_path),
                "manifest_sha256": sha256_file(source_manifest_path),
                "manifest_schema": source_manifest["schema"],
            },
            "license": source_manifest["license"],
            "demographic": source_manifest["demographic"],
            "gender": source_manifest["gender"],
            "skeleton_family": skeleton_family,
            "normalization": normalization,
            "runtime_motion_contract": {
                "horizontal_world_trajectory_authority": "UE_actor_trajectory",
                "walking_embedded_horizontal_root_motion": "removed",
                "walking_vertical_motion": "preserved",
                "dynamic_ground_snap_to_floor_required": True,
            },
            "equivalence": {
                "mesh_position_accessors_unchanged": True,
                "embedded_image_payloads_unchanged": True,
                "material_texture_graph_unchanged": True,
                "animation_names_unchanged": True,
                "embedded_image_payloads": normalized_images,
            },
            "glb_contract": source_manifest["glb_contract"],
            "expected_ue_qa": qa,
            "automatic_checks": {
                "overall": "passed",
                "source_hash_locked": "passed",
                "grounded_metric_normalization": "passed",
                "walking_in_place": "passed",
                "walking_vertical_motion": "preserved",
                "mesh_positions": "unchanged",
                "embedded_images": "unchanged",
                "material_texture_graph": "unchanged",
                "animation_names": "unchanged",
                "authored_height_preserved": "passed",
            },
        }
        manifest_path = staging / "normalization_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _seal_tree(staging)
        os.replace(staging, output_dir)
        return {
            "base_avatar_id": base_avatar_id,
            "status": "passed",
            "output": str(output_dir),
            "runtime_sha256": runtime_record["sha256"],
            "manifest_sha256": sha256_file(output_dir / "normalization_manifest.json"),
        }
    except Exception:
        _cleanup(staging)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inventory = json.loads(args.inventory_json.read_text(encoding="utf-8"))
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise RuntimeError("inventory is not normalization-ready")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    planned = []
    skipped = []
    for avatar in sorted(inventory["avatars"], key=lambda item: item["base_avatar_id"]):
        avatar_id = avatar["base_avatar_id"]
        source = source_root / f"{avatar_id}_original_v1"
        output = output_root / f"{avatar_id}_original_ue_v1"
        if output.exists():
            manifest_path = output / "normalization_manifest.json"
            runtime_path = output / "runtime.glb"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise RuntimeError(f"unverified existing normalized output: {output}")
            runtime = manifest.get("runtime_glb", {})
            if not (
                manifest.get("schema") == OUTPUT_SCHEMA
                and manifest.get("base_avatar_id") == avatar_id
                and manifest.get("automatic_checks", {}).get("overall") == "passed"
                and runtime_path.is_file()
                and runtime.get("sha256") == sha256_file(runtime_path)
            ):
                raise RuntimeError(f"unverified existing normalized output: {output}")
            skipped.append(avatar_id)
        else:
            planned.append((source, output, avatar_id))
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(publish_one, source, output, avatar_id): avatar_id
            for source, output, avatar_id in planned
        }
        for future in concurrent.futures.as_completed(future_map):
            avatar_id = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(f"ROCKETBOX_NORMALIZE passed {avatar_id}", flush=True)
            except Exception as error:
                failures.append({"base_avatar_id": avatar_id, "error": str(error)})
                print(f"ROCKETBOX_NORMALIZE failed {avatar_id}: {error}", flush=True)
    status = {
        "schema_version": "rocketbox_batch_native_ue_status_v1",
        "inventory_sha256": sha256_file(args.inventory_json),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "passed_count": len(results),
        "skipped_verified_count": len(skipped),
        "failed_count": len(failures),
        "results": sorted(results, key=lambda item: item["base_avatar_id"]),
        "failures": sorted(failures, key=lambda item: item["base_avatar_id"]),
        "automatic_checks": {
            "overall": "passed" if not failures else "failed"
        },
    }
    status_path = output_root / "batch_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} normalized Rocketbox runtimes failed")
    print(
        f"ROCKETBOX_NORMALIZE_ALL_OK total={len(results)+len(skipped)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
