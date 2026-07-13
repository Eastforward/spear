#!/usr/bin/env python3
"""Publish immutable in-place, metric Rocketbox runtimes for Unreal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import build_ue_normalized_rocketbox_runtime as v2
from tools.normalize_rocketbox_glb_for_ue import (
    normalize_in_place_grounded_metric_glb_bytes,
    read_glb_bytes,
)


CONTRACTS = {
    "rocketbox_female_adult_01_original_ue_v3": {
        "asset_id": "rocketbox_female_adult_01",
        "variant_id": "original_v1",
        "source_tag": "rocketbox_female_adult_01_original_v1",
        "source_relative_root": (
            "tmp/rocketbox_native_runtime_v1/"
            "rocketbox_female_adult_01_original_v1"
        ),
        "source_manifest": "build_manifest.json",
        "source_manifest_schema": "rocketbox_native_runtime_build_v1",
        "output_relative_root": (
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_female_adult_01_original_ue_v3"
        ),
    },
    "rocketbox_male_adult_01_original_ue_v3": {
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
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_male_adult_01_original_ue_v3"
        ),
    },
    "rocketbox_male_adult_01_shirt_blue_ue_v3": {
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
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_male_adult_01_shirt_blue_ue_v3"
        ),
    },
}


class UeInPlaceRuntimeBundleError(RuntimeError):
    """Raised when a v3 immutable runtime contract cannot be published."""


def _cleanup(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def publish_bundle(tag: str, contract: dict[str, object], output: Path) -> Path:
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise UeInPlaceRuntimeBundleError(
            f"refusing to replace UE runtime bundle: {output}"
        )
    try:
        source_root, source_glb, source_manifest_path, source_manifest = (
            v2._load_source(tag, contract)
        )
    except v2.UeRuntimeBundleError as error:
        raise UeInPlaceRuntimeBundleError(str(error)) from error

    source_payload = source_glb.read_bytes()
    normalized_payload, normalization = (
        normalize_in_place_grounded_metric_glb_bytes(source_payload)
    )
    source_document, source_binary = read_glb_bytes(source_payload)
    normalized_document, normalized_binary = read_glb_bytes(normalized_payload)
    source_images = v2._image_payloads(source_document, source_binary)
    normalized_images = v2._image_payloads(normalized_document, normalized_binary)
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
    walking_motion = normalization.get("root_motion", {}).get("Walking", {})
    displacement = walking_motion.get("horizontal_displacement_before_m")
    if (
        source_images != normalized_images
        or not material_graph_unchanged
        or source_animation_names != normalized_animation_names
        or set(normalized_animation_names) != {"Walking", "Standing_Idle"}
        or normalization.get("normalized_joint_count") != 80
        or normalization.get("static_wrapper_translation_zeroed") is not True
        or normalization.get("in_place_actions") != ["Walking"]
        or not isinstance(displacement, list)
        or len(displacement) != 2
        or sum(float(value) ** 2 for value in displacement) ** 0.5 < 1.4
        or float(walking_motion.get("maximum_horizontal_deviation_after_m", 1.0))
        >= 1.0e-6
        or float(walking_motion.get("maximum_vertical_world_error_m", 1.0))
        >= 1.0e-6
    ):
        raise UeInPlaceRuntimeBundleError(
            "in-place normalized runtime equivalence checks failed"
        )

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
            raise UeInPlaceRuntimeBundleError("normalized runtime readback changed")
        runtime_record = {
            "filename": "runtime.glb",
            "size_bytes": runtime_path.stat().st_size,
            "sha256": v2._sha256_file(runtime_path),
        }
        manifest = {
            "schema": "rocketbox_native_ue_runtime_v3",
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
                "runtime_glb_sha256": v2._sha256_file(source_glb),
                "runtime_glb_size_bytes": source_glb.stat().st_size,
                "manifest": str(source_manifest_path),
                "manifest_sha256": v2._sha256_file(source_manifest_path),
                "manifest_schema": source_manifest["schema"],
            },
            "license": source_manifest.get("license"),
            "normalization": normalization,
            "runtime_motion_contract": {
                "horizontal_world_trajectory_authority": "UE_actor_trajectory",
                "walking_embedded_horizontal_root_motion": "removed",
                "walking_vertical_motion": "preserved",
                "dynamic_ground_snap_to_floor_required": True,
            },
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
                "ground_snap_to_floor": True,
                "ground_snap_max_abs_correction_cm": 15.0,
                "ground_snap_residual_tolerance_cm": 0.1,
            },
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
            raise UeInPlaceRuntimeBundleError(
                "normalized runtime inventory changed"
            )
        v2._seal_tree(staging)
        if output.exists() or output.is_symlink():
            raise UeInPlaceRuntimeBundleError(
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
                "manifest_sha256": v2._sha256_file(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
