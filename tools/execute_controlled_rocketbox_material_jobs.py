#!/usr/bin/env python3
"""Execute deduplicated Rocketbox material jobs from an authenticated preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_source_asset_execution as preparation
from tools import rocketbox_native_material_canary as native


ATTEMPT_SCHEMA = "avengine_controlled_rocketbox_material_attempt_v1"
BATCH_SCHEMA = "avengine_controlled_rocketbox_material_batch_v1"
SUPPORTED_BASE_AVATAR = "rocketbox_adults_male_adult_01"
SUPPORTED_MASK = "shirt_main_color"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
NATIVE_RUNTIME_ROOT = SPEAR_ROOT / "tmp/rocketbox_native_runtime_v1"
NATIVE_RUNTIME_FILENAME = "runtime.glb"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(f"artifact escapes output root: {path}") from error
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_bytes_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise contracts.ContractError(f"refusing to replace existing artifact: {path}") from error


def _save_png_no_replace(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise contracts.ContractError(f"refusing to replace existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            Image.fromarray(values, mode="RGB").save(
                handle, format="PNG", optimize=False, compress_level=9
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_preflight(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"preflight is missing or unsafe: {path}")
    preflight = preparation.validate_execution_preflight(contracts.load_json(path))
    roots = {
        root_id: Path(root_path).resolve()
        for root_id, root_path in preflight["artifact_roots"].items()
    }
    rebuilt = preparation.build_execution_preflight(
        Path(preflight["source_bundle"]["input_dir"]), roots
    )
    if contracts.canonical_json(preflight) != contracts.canonical_json(rebuilt):
        raise contracts.ContractError(
            "execution preflight no longer matches current authenticated inputs"
        )
    return preflight


def validate_material_job(job: Any) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise contracts.ContractError("Rocketbox material job must be an object")
    if job.get("base_avatar_id") != SUPPORTED_BASE_AVATAR:
        raise contracts.ContractError(
            f"material executor has no audited adapter for {job.get('base_avatar_id')!r}"
        )
    plan = job.get("material_edit_plan")
    if not isinstance(plan, dict):
        raise contracts.ContractError("material edit plan is missing")
    if (
        plan.get("schema") != "rocketbox_material_edit_plan_v1"
        or plan.get("route") != "rocketbox_material_v1"
        or plan.get("base_avatar_id") != SUPPORTED_BASE_AVATAR
        or plan.get("geometry_changes_allowed") is not False
        or plan.get("flux_texture_detail")
        != {"enabled": False, "policy": "approved_mask_optional_only"}
    ):
        raise contracts.ContractError("Rocketbox material edit plan contract changed")
    edits = plan.get("edits")
    if not isinstance(edits, list) or len(edits) != 1:
        raise contracts.ContractError("current audited Rocketbox adapter requires one edit")
    edit = edits[0]
    required = {
        "attribute",
        "value",
        "semantic_mask",
        "mask_registry",
        "source_texture_role",
        "operation",
        "target_srgb_u8",
        "target_srgb_hex",
    }
    if not isinstance(edit, dict) or set(edit) != required:
        raise contracts.ContractError("material edit fields changed")
    if (
        edit["attribute"] != "top_color"
        or edit["semantic_mask"] != SUPPORTED_MASK
        or edit["source_texture_role"] != "body_base_color"
        or edit["operation"] != "replace_base_color_preserve_pbr_detail_v1"
        or job.get("sampled_attributes") != {"top_color": edit["value"]}
    ):
        raise contracts.ContractError("material edit is outside the audited shirt contract")
    color = edit["target_srgb_u8"]
    if (
        not isinstance(color, list)
        or len(color) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in color
        )
        or edit["target_srgb_hex"]
        != "#" + "".join(f"{channel:02X}" for channel in color)
    ):
        raise contracts.ContractError("material target color is invalid")
    consumers = job.get("consumer_requests")
    if not isinstance(consumers, list) or not consumers:
        raise contracts.ContractError("material job has no consumer requests")
    return copy.deepcopy(job)


def _authenticate_mask_registry(
    edit: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    roots = {
        root_id: Path(path) for root_id, path in preflight["artifact_roots"].items()
    }
    record = input_builder.authenticate_artifact_record(
        edit["mask_registry"],
        roots,
        role="material_mask_registry:top_color",
        owner="controlled Rocketbox material executor",
    )
    root = roots[edit["mask_registry"]["root_id"]].resolve()
    path = (root / edit["mask_registry"]["path"]).resolve()
    registry = contracts.load_json(path)
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != "rocketbox_native_semantic_mask_registry_v1"
        or registry.get("asset_id") != native.ASSET_ID
        or registry.get("source_body_color_sha256")
        != native.SOURCE_BODY_COLOR_SHA256
        or registry.get("face_count") != native.SHIRT_FACE_COUNT
        or registry.get("face_indices_u32le_sha256")
        != native.SHIRT_FACE_INDICES_U32LE_SHA256
        or registry.get("masks") != native.FROZEN_MASK_REGISTRY
    ):
        raise contracts.ContractError("registered Rocketbox shirt mask contract changed")
    return {**record, "resolved_path": str(path)}


def _build_attempt(
    job: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source: Mapping[str, Any],
    masks: Mapping[str, Any],
    destination: Path,
    batch_root: Path,
    public_root: Path,
) -> dict[str, Any]:
    job = validate_material_job(job)
    edit = job["material_edit_plan"]["edits"][0]
    mask_authentication = _authenticate_mask_registry(edit, preflight)
    variant = native.build_tga_variant(source, masks, edit["target_srgb_u8"])
    if (
        variant["qa"]["outside_mask_changed_pixels"] != 0
        or variant["qa"]["protected_changed_pixels"] != 0
        or variant["qa"]["header_changed_bytes"] != 0
        or variant["qa"]["footer_changed_bytes"] != 0
        or variant["qa"]["size_unchanged"] is not True
        or float(variant["qa"]["linear_luminance_correlation"]) < 0.98
    ):
        raise contracts.ContractError("material output failed protected-pixel QA")

    variant_dir = destination / "variant"
    diagnostics_dir = destination / "diagnostics"
    variant_dir.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)
    texture_path = variant_dir / "m002_body_color.tga"
    _write_bytes_no_replace(texture_path, variant["tga_bytes"])
    if _sha256_file(texture_path) != variant["output_sha256"]:
        raise contracts.ContractError("material texture writeback hash mismatch")
    diff = native._build_texture_diff(source["tga"]["rgb"], variant["rgb"])
    diff_path = diagnostics_dir / "texture_diff.png"
    _save_png_no_replace(diff_path, diff)

    texture_record = _file_record(texture_path, root=batch_root)
    variant_request = preparation.build_rocketbox_runtime_variant_request(
        job, texture_record
    )
    request_path = destination / "variant_request.json"
    contracts.write_json_no_replace(request_path, variant_request)
    request_record = _file_record(request_path, root=batch_root)
    runtime_asset_id = variant_request["asset_id"]
    runtime_tag = variant_request["tag"]
    final_destination = public_root / destination.relative_to(batch_root)
    texture_final = final_destination / "variant/m002_body_color.tga"
    request_final = final_destination / "variant_request.json"
    blender_command = [
        str(native.BLENDER_PATH),
        "--background",
        "--factory-startup",
        "--python",
        str(
            Path(__file__).resolve().parent
            / "blender_build_native_rocketbox_runtime.py"
        ),
        "--",
        "--body-color-texture",
        str(texture_final),
        "--variant-manifest",
        str(request_final),
    ]
    attempt: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "status": "passed",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "execution_job_id": job["execution_job_id"],
        "variant_key": job["variant_key"],
        "variant_id": job["variant_id"],
        "runtime_asset_id": runtime_asset_id,
        "runtime_tag": runtime_tag,
        "profile_schema_id": job["profile_schema_id"],
        "profile_sha256": job["profile_sha256"],
        "lineage_group_id": job["lineage_group_id"],
        "sampled_attributes": job["sampled_attributes"],
        "consumer_requests": job["consumer_requests"],
        "material_edit_plan": job["material_edit_plan"],
        "inputs": {
            "execution_preflight_sha256": preflight["preflight_sha256"],
            "base_template": job["base_template"],
            "mask_registry_authentication": mask_authentication,
            "source_body_color": source["body_color"],
        },
        "outputs": {
            "body_color_texture": texture_record,
            "variant_request": request_record,
            "texture_diff": _file_record(diff_path, root=batch_root),
        },
        "material_qa": variant["qa"],
        "runtime_handoff": {
            "status": "pending",
            "builder": "blender_build_native_rocketbox_runtime.py",
            "command": blender_command,
            "expected_output": str(
                NATIVE_RUNTIME_ROOT / runtime_tag / NATIVE_RUNTIME_FILENAME
            ),
            "expected_actions": ["Walking", "Standing_Idle"],
            "expected_front_axis": "negative_y",
            "actor_scale": 1.0,
        },
        "automatic_checks": {
            "preflight_reauthenticated": True,
            "mask_registry_authenticated": True,
            "geometry_changes_allowed": False,
            "inside_mask_changed_pixels": variant["qa"][
                "inside_mask_changed_pixels"
            ],
            "outside_mask_changed_pixels": 0,
            "protected_detail_changed_pixels": 0,
            "tga_container_unchanged": True,
            "overall": "passed",
        },
    }
    attempt["attempt_sha256"] = contracts.manifest_sha256(attempt)
    attempt_path = destination / "attempt_manifest.json"
    contracts.write_json_no_replace(attempt_path, attempt)
    return {
        "variant_key": job["variant_key"],
        "variant_id": job["variant_id"],
        "runtime_tag": runtime_tag,
        "consumer_request_count": len(job["consumer_requests"]),
        "sampled_attributes": job["sampled_attributes"],
        "attempt_manifest": _file_record(attempt_path, root=batch_root),
        "body_color_texture": texture_record,
        "variant_request": request_record,
        "runtime_handoff_status": "pending",
    }


def execute_material_jobs(
    preflight: Mapping[str, Any], output_root: Path
) -> Path:
    preflight = preparation.validate_execution_preflight(preflight)
    jobs = [validate_material_job(job) for job in preflight["routes"]["rocketbox_material_v1"]]
    if not jobs:
        raise contracts.ContractError("preflight contains no Rocketbox material jobs")
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(
            f"refusing to replace existing output directory: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        source = native.load_authenticated_source()
        uv_bundle = native.extract_authenticated_shirt_uv(source)
        masks = native.build_shirt_masks(source, uv_bundle)
        if masks["mask_sha256"] != {
            name: record["raw_bool_sha256"]
            for name, record in native.FROZEN_MASK_REGISTRY.items()
        }:
            raise contracts.ContractError("fresh Rocketbox shirt masks changed")
        variants_dir = staging / "variants"
        variants_dir.mkdir()
        results = []
        for job in jobs:
            destination = variants_dir / job["variant_id"]
            destination.mkdir()
            results.append(
                _build_attempt(
                    job,
                    preflight,
                    source,
                    masks,
                    destination,
                    staging,
                    output_root,
                )
            )
        batch: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "execution_preflight_sha256": preflight["preflight_sha256"],
            "planned_request_count": sum(
                len(job["consumer_requests"]) for job in jobs
            ),
            "unique_variant_count": len(results),
            "deduplicated_request_count": sum(
                len(job["consumer_requests"]) for job in jobs
            )
            - len(results),
            "variants": sorted(results, key=lambda item: item["variant_key"]),
            "automatic_checks": {
                "preflight_reauthenticated": True,
                "unique_variant_keys": len({item["variant_key"] for item in results})
                == len(results),
                "all_material_attempts_passed": True,
                "baseline_written": False,
                "overall": "passed",
            },
        }
        batch["batch_sha256"] = contracts.manifest_sha256(batch)
        contracts.write_json_no_replace(staging / "material_batch_manifest.json", batch)
        native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return output_root / "material_batch_manifest.json"
    except Exception:
        native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        preflight = _load_preflight(args.preflight)
        manifest = execute_material_jobs(preflight, args.output_root)
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, native.NativeMaterialCanaryError) as error:
        print(f"CONTROLLED_ROCKETBOX_MATERIAL_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ROCKETBOX_MATERIAL_OK "
        f"variants={payload['unique_variant_count']} "
        f"requests={payload['planned_request_count']} "
        f"deduplicated={payload['deduplicated_request_count']} "
        f"output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
