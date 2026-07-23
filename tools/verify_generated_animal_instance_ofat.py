#!/usr/bin/env python3
"""Verify an emitted nine-case generated-animal OFAT batch without Blender.

This verifier authenticates the immutable GLBs and their per-instance
manifests, checks the one-factor-at-a-time attribute matrix, reads each GLB
container back, and follows the two non-baseline coat records to their
undistilled FLUX real-reference edit and spatial UV-projection evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct


BATCH_SCHEMA = "avengine_generated_animal_instance_ofat_batch_v2"
INSTANCE_SCHEMA = "avengine_generated_animal_instance_ofat_v2"
VERIFY_SCHEMA = "avengine_generated_animal_instance_ofat_verification_v1"
EXPECTED_VARIANTS = {
    "baseline": ("medium", "standard", "black_white", "adult", None),
    "size_small": ("small", "standard", "black_white", "adult", "size"),
    "size_large": ("large", "standard", "black_white", "adult", "size"),
    "build_slim": ("medium", "slim", "black_white", "adult", "body_build"),
    "build_stocky": ("medium", "stocky", "black_white", "adult", "body_build"),
    "coat_blue_merle": ("medium", "standard", "blue_merle", "adult", "coat"),
    "coat_red_white": ("medium", "standard", "red_white", "adult", "coat"),
    "age_young": ("medium", "standard", "black_white", "young", "life_stage"),
    "age_senior": ("medium", "standard", "black_white", "senior", "life_stage"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or unsafe {label}: {path}")
    return path


def load_json(path: Path, label: str) -> dict:
    path = require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def glb_document(path: Path) -> dict:
    raw = require_file(path, "instance GLB").read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError(f"invalid GLB header: {path}")
    version, declared_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_length != len(raw):
        raise ValueError(f"invalid GLB version or length: {path}")
    offset = 12
    document = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError(f"truncated GLB chunk header: {path}")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset : offset + length]
        offset += length
        if len(payload) != length:
            raise ValueError(f"truncated GLB chunk: {path}")
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b"\x00 ").decode("utf-8"))
    if offset != len(raw) or not isinstance(document, dict):
        raise ValueError(f"invalid GLB chunk layout: {path}")
    return document


def verify_glb(path: Path) -> dict:
    document = glb_document(path)
    animations = sorted(item.get("name") for item in document.get("animations", []))
    primitives = [
        primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    skinned = [
        primitive
        for primitive in primitives
        if {"JOINTS_0", "WEIGHTS_0"}.issubset(primitive.get("attributes", {}))
    ]
    materials = document.get("materials", [])
    textured = [
        material
        for material in materials
        if "baseColorTexture" in material.get("pbrMetallicRoughness", {})
    ]
    if animations != ["Idle", "Walking"]:
        raise ValueError(f"Walk/Idle readback failed for {path}: {animations}")
    if len(document.get("skins", [])) != 1 or len(skinned) != 1:
        raise ValueError(f"single-skin contract failed for {path}")
    if len(textured) != len(materials) or not materials:
        raise ValueError(f"Base Color texture contract failed for {path}")
    return {
        "animations": animations,
        "skin_count": 1,
        "skinned_primitive_count": 1,
        "base_color_texture_material_count": len(materials),
    }


def finite_vector(value, length: int, label: str):
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ValueError(f"invalid {label}: {value}")


def verify_coat_evidence(instance: dict, baseline_input: Path) -> dict:
    appearance = instance.get("appearance", {})
    coat = instance["attributes"]["coat"]
    board = require_file(Path(appearance.get("appearance_reference_board", "")), "appearance board")
    if appearance.get("not_global_rgb_factor") is not True:
        raise ValueError(f"coat {coat} does not reject global RGB scaling")
    if coat == "black_white":
        source = load_json(Path(appearance.get("manifest", "")), "baseline generation manifest")
        model = source.get("model", {})
        if (
            appearance.get("method") != "real_reference_flux_source_asset_generation"
            or not isinstance(model, dict)
            or model.get("is_distilled") is not False
            or model.get("execution") != "full_single_gpu_no_cpu_offload"
        ):
            raise ValueError("baseline coat lacks undistilled direct-GPU FLUX evidence")
        return {"coat": coat, "method": appearance["method"], "appearance_board": str(board)}

    if appearance.get("method") != "real_reference_flux_multiview_edit_then_uv_projection":
        raise ValueError(f"coat {coat} lacks real-reference FLUX projection evidence")
    flux = load_json(Path(appearance.get("flux_edit_manifest", "")), f"{coat} FLUX manifest")
    projection = load_json(
        Path(appearance.get("projection_manifest", "")), f"{coat} projection manifest"
    )
    if (
        not str(flux.get("schema", "")).startswith(
            "avengine_flux2_base_animal_multiview_coat_edit_v"
        )
        or flux.get("is_distilled") is not False
        or flux.get("one_model_invocation") is not True
        or flux.get("reference_image_count") != 2
        or flux.get("geometry_rig_or_animation_edit_authorized") is not False
        or flux.get("negative_conditioning")
        != "native_negative_prompt_embeddings_with_cfg"
        or flux.get("appearance_reference_board") != str(board)
    ):
        raise ValueError(f"coat {coat} FLUX evidence is incomplete")
    if (
        projection.get("schema")
        != "avengine_generated_animal_multiview_coat_projection_v2"
        or projection.get("not_global_rgb_factor") is not True
        or projection.get("geometry_skin_skeleton_and_actions_preserved_by_design")
        is not True
        or Path(projection.get("input_glb", "")).resolve() != baseline_input
        or Path(projection.get("output_glb", "")).resolve()
        != Path(instance.get("source_glb", "")).resolve()
        or projection.get("container_patch", {}).get("protected_json_sections_unchanged")
        != ["nodes", "meshes", "skins", "accessors", "animations"]
    ):
        raise ValueError(f"coat {coat} projection did not preserve asset authority")
    return {
        "coat": coat,
        "method": appearance["method"],
        "appearance_board": str(board),
        "flux_model": flux.get("model"),
        "flux_revision": flux.get("revision"),
        "projection_method": projection.get("projection_method"),
        "direct_coverage_ratio": projection.get("direct_coverage_ratio"),
    }


def main(argv=None):
    args = parse_args(argv)
    batch_path = require_file(args.batch_manifest, "batch manifest")
    batch = load_json(batch_path, "batch manifest")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace output: {output}")
    if (
        batch.get("schema") != BATCH_SCHEMA
        or batch.get("state_classification") != "research_candidate"
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("breed") != "border_collie"
        or batch.get("variant_count") != 9
        or set(batch.get("coat_ids", [])) != {"black_white", "blue_merle", "red_white"}
    ):
        raise ValueError("batch contract mismatch")
    expected_domains = {
        "size": {"small", "medium", "large"},
        "body_build": {"slim", "standard", "stocky"},
        "life_stage": {"young", "adult", "senior"},
        "coat": {"black_white", "blue_merle", "red_white"},
    }
    for key, values in expected_domains.items():
        if set(batch.get("attribute_domains", {}).get(key, [])) != values:
            raise ValueError(f"attribute domain mismatch: {key}")

    baseline_input = require_file(Path(batch.get("baseline_input_glb", "")), "baseline GLB")
    records = {}
    coat_evidence = {}
    for summary in batch.get("results", []):
        variant = summary.get("variant_id")
        if variant not in EXPECTED_VARIANTS or variant in records:
            raise ValueError(f"unexpected or duplicate variant: {variant}")
        manifest_path = batch_path.parent / variant / "manifest.json"
        instance = load_json(manifest_path, f"{variant} manifest")
        size, build, coat, life_stage, changed = EXPECTED_VARIANTS[variant]
        expected_attributes = {
            "breed": "border_collie",
            "size": size,
            "body_build": build,
            "coat": coat,
            "life_stage": life_stage,
        }
        if (
            instance.get("schema") != INSTANCE_SCHEMA
            or instance.get("variant_id") != variant
            or instance.get("attributes") != expected_attributes
            or summary.get("attributes") != expected_attributes
            or instance.get("changed_attribute_from_baseline") != changed
            or summary.get("changed_attribute_from_baseline") != changed
            or instance.get("state_classification") != "research_candidate"
            or instance.get("formal_dataset_registration_authorized") is not False
            or instance.get("actual_generated_mesh_preserved") is not True
            or instance.get("template_geometry_used") is not False
            or instance.get("global_rgb_material_factor_used") is not False
            or instance.get("walk_idle_preserved") is not True
        ):
            raise ValueError(f"instance contract mismatch: {variant}")
        glb = require_file(Path(instance.get("output_glb", "")), f"{variant} GLB")
        digest = sha256_file(glb)
        if (
            digest != instance.get("output_sha256")
            or digest != summary.get("output_sha256")
            or glb != Path(summary.get("output_glb", "")).resolve()
        ):
            raise ValueError(f"GLB authentication failed: {variant}")
        readback = verify_glb(glb)
        if readback != instance.get("readback") or readback != summary.get("readback"):
            raise ValueError(f"GLB readback changed since emission: {variant}")
        grounding = instance.get("grounding", {})
        if (
            grounding.get("passed") is not True
            or abs(float(grounding.get("rest_minimum_z_after_m", math.inf)))
            > float(grounding.get("tolerance_m", 0.0))
        ):
            raise ValueError(f"grounding failed: {variant}")
        emitter = instance.get("emitter_anchor", {})
        finite_vector(emitter.get("emitter_offset_m"), 3, f"{variant} emitter")
        finite_vector(emitter.get("local_forward_axis"), 3, f"{variant} forward axis")
        if (
            emitter.get("method") != "semantic_head_forward_quantile_rest_mesh_v1"
            or emitter.get("asset_specific_not_species_template") is not True
            or emitter.get("mouth_animation_required") is not False
        ):
            raise ValueError(f"emitter contract failed: {variant}")
        evidence = verify_coat_evidence(instance, baseline_input)
        coat_evidence.setdefault(coat, evidence)
        records[variant] = {
            "attributes": expected_attributes,
            "output_glb": str(glb),
            "output_sha256": digest,
            "grounding_minimum_z_m": grounding["rest_minimum_z_after_m"],
            "emitter_offset_m": emitter["emitter_offset_m"],
            "readback": readback,
        }
    if set(records) != set(EXPECTED_VARIANTS):
        raise ValueError("nine-case OFAT matrix is incomplete")

    manifests = {
        key: load_json(batch_path.parent / key / "manifest.json", f"{key} manifest")
        for key in records
    }
    if not (
        manifests["size_small"]["size_ratio"] < manifests["baseline"]["size_ratio"]
        < manifests["size_large"]["size_ratio"]
    ):
        raise ValueError("size order is not small < medium < large")
    emitter_size_order = {
        "small": records["size_small"]["emitter_offset_m"],
        "medium": records["baseline"]["emitter_offset_m"],
        "large": records["size_large"]["emitter_offset_m"],
    }
    for axis in (0, 1):
        if not (
            emitter_size_order["small"][axis]
            < emitter_size_order["medium"][axis]
            < emitter_size_order["large"][axis]
        ):
            raise ValueError(
                "size-specific emitter did not scale in final asset-root space"
            )
    slim = manifests["build_slim"]["shape"]["semantic_measurements"][
        "torso_weighted_lateral_vertical_rms_ratio"
    ]
    stocky = manifests["build_stocky"]["shape"]["semantic_measurements"][
        "torso_weighted_lateral_vertical_rms_ratio"
    ]
    young = manifests["age_young"]["shape"]["semantic_measurements"][
        "head_weighted_radius_rms_ratio"
    ]
    senior = manifests["age_senior"]["shape"]["semantic_measurements"][
        "head_weighted_radius_rms_ratio"
    ]
    if not (slim < 1.0 < stocky and senior < 1.0 < young):
        raise ValueError("body-build or life-stage measured order failed")
    senior_surface = manifests["age_senior"].get("life_stage_surface", {})
    if (
        senior_surface.get("method") != "semantic_uv_muzzle_neutral_gray_floor_v1"
        or senior_surface.get("age_surface_pixels_modified") is not True
        or manifests["age_senior"].get("global_rgb_material_factor_used") is not False
    ):
        raise ValueError("senior cue is not a local semantic surface edit")

    verification = {
        "schema": VERIFY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_research_candidate_instance_validation",
        "formal_dataset_registration_authorized": False,
        "batch_manifest": {
            "path": str(batch_path),
            "sha256": sha256_file(batch_path),
        },
        "verified_variant_count": len(records),
        "verified_attribute_domains": {
            key: sorted(values) for key, values in expected_domains.items()
        },
        "measured_orders": {
            "size_ratio": {
                "small": manifests["size_small"]["size_ratio"],
                "medium": manifests["baseline"]["size_ratio"],
                "large": manifests["size_large"]["size_ratio"],
            },
            "size_specific_emitter_offset_m": emitter_size_order,
            "body_build_torso_ratio": {"slim": slim, "standard": 1.0, "stocky": stocky},
            "life_stage_head_ratio": {"senior": senior, "adult": 1.0, "young": young},
        },
        "coat_evidence": coat_evidence,
        "variants": records,
        "claims": {
            "all_glbs_hash_authenticated": True,
            "all_glbs_read_back_with_idle_walking_skin_and_base_color": True,
            "all_instances_grounded": True,
            "all_emitters_asset_specific": True,
            "nonbaseline_coats_use_real_reference_undistilled_flux": True,
            "nonbaseline_coats_use_spatial_uv_projection_not_global_rgb": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"GENERATED_ANIMAL_INSTANCE_OFAT_VERIFY_OK variants=9 output={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"GENERATED_ANIMAL_INSTANCE_OFAT_VERIFY_FAILED {error}")
        raise SystemExit(2)
