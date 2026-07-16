#!/usr/bin/env python3
"""Run one-attribute-at-a-time FLUX.2 edits from approved soft-T references."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from tools import human_attribute_masks as semantic_masks
from tools import route2_human_contract_common as route2_common
from tools import route2_human_qualified_candidate as qualified_candidate
from tools.spike_rlr import human_attribute_review as attribute_review
from tools.spike_rlr import tokenrig_human_review as route2_human_review


MODEL_ROOT = Path("/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B")
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
JOBS_SCHEMA = "flux2_human_attribute_jobs_v2"
OUTPUT_SCHEMA = "flux2_human_attribute_candidate_v2"
WIDTH = 1152
HEIGHT = 1536
STEPS = 28
GUIDANCE_SCALE = 1.0
MAX_SEQUENCE_LENGTH = 512
PHYSICAL_GPU = "3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
RUNNER_PATH = Path(__file__).resolve()
ROUTE2_OUTPUT_ROOT = RUNNER_PATH.parents[1] / "tmp/pixal_tokenrig_route2_v1"
MODEL_INVENTORY_PATH = (
    RUNNER_PATH.parents[1]
    / "tmp/human_attribute_instances_v1/flux2_snapshot_inventory_v1.json"
)
MODEL_INVENTORY_SHA256 = "962ec618f2846728da8ac4ccb18fb61bdf6334c729017b3feaa48ae7710f04a4"
ISNET_MODEL_PATH = Path("/data/models/rembg/isnet-general-use/isnet-general-use.onnx")
ISNET_MODEL_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
ISNET_PROVENANCE_PATH = (
    RUNNER_PATH.parents[1]
    / "tmp/human_attribute_instances_v1/isnet_provenance_v1.json"
)
ISNET_PROVENANCE_SHA256 = "42db586046ba2d11cac285074085439037fb6036b8fc294cbbe291dceedbb798"


CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    case_id: {
        **dict(contract),
        "seed": 101 + index,
    }
    for index, (case_id, contract) in enumerate(
        semantic_masks.CASE_MASK_CONTRACTS.items()
    )
}

REQUIRED_JOB_FIELDS = frozenset(
    {
        "case_id",
        "base_asset_id",
        "base_qualified_candidate",
        "downstream_asset_id",
        "source_image",
        "source_image_sha256",
        "source_review",
        "source_review_sha256",
        "source_candidate_manifest",
        "source_alpha",
        "source_rgba",
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "model",
        "runner_sha256",
        "mask_construction_version",
        "mask_bundle",
        "mask_decision",
        "target_parameters",
        "isnet",
    }
)


def sha256_file(path: Path) -> str:
    return route2_common.sha256_file(Path(path))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: Path, description: str) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"{description} must be a direct regular file: {path}")
    return path


def _validate_hash(value: Any, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_path_hash_record(value: Any, description: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{description} must contain exactly path and sha256")
    if not isinstance(value.get("path"), str) or not value["path"].strip():
        raise ValueError(f"{description} path must be non-empty")
    _validate_hash(value.get("sha256"), f"{description} sha256")
    return value


def validate_job(job: Mapping[str, Any]) -> None:
    missing = REQUIRED_JOB_FIELDS - set(job)
    extra = set(job) - REQUIRED_JOB_FIELDS
    if missing or extra:
        raise ValueError(f"job fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    case_id = job.get("case_id")
    if case_id not in CASE_CONTRACTS:
        raise ValueError(f"unknown attribute case: {case_id!r}")
    contract = CASE_CONTRACTS[str(case_id)]
    if job.get("base_asset_id") != contract["base_asset_id"]:
        raise ValueError(f"base asset mismatch for {case_id}")
    canonical_pointer = (
        ROUTE2_OUTPUT_ROOT
        / str(job["base_asset_id"])
        / qualified_candidate.FILENAME
    )
    if job.get("base_qualified_candidate") != str(canonical_pointer):
        raise ValueError(f"base qualified candidate path mismatch for {case_id}")
    if job.get("seed") != contract["seed"]:
        raise ValueError(f"seed mismatch for {case_id}")
    if job.get("downstream_asset_id") != f"route2_{case_id}_v1":
        raise ValueError(f"downstream asset mismatch for {case_id}")
    if job.get("mask_construction_version") != semantic_masks.MASK_CONSTRUCTION_VERSION:
        raise ValueError(f"mask construction version mismatch for {case_id}")
    if job.get("target_parameters") != contract["target_parameters"]:
        raise ValueError(f"target parameters mismatch for {case_id}")
    if (job.get("width"), job.get("height")) != (WIDTH, HEIGHT):
        raise ValueError(f"attribute jobs must be exactly {WIDTH}x{HEIGHT}")
    if job.get("steps") != STEPS or float(job.get("guidance_scale", -1)) != GUIDANCE_SCALE:
        raise ValueError("generation steps/guidance differ from the pinned route-2 contract")
    for key in ("source_image_sha256", "source_review_sha256", "runner_sha256"):
        _validate_hash(job.get(key), key)
    if job.get("runner_sha256") != sha256_file(RUNNER_PATH):
        raise ValueError("attribute runner hash differs from the job contract")
    for key in ("source_image", "source_review", "prompt", "negative_prompt"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in (
        "source_candidate_manifest",
        "source_alpha",
        "source_rgba",
        "mask_bundle",
        "mask_decision",
    ):
        _validate_path_hash_record(job.get(key), key)
    expected_model = {
        "name": "black-forest-labs/FLUX.2-klein-4B",
        "root": str(MODEL_ROOT),
        "revision": MODEL_REVISION,
        "inventory": str(MODEL_INVENTORY_PATH),
        "inventory_sha256": MODEL_INVENTORY_SHA256,
        "local_files_only": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }
    if job.get("model") != expected_model:
        raise ValueError("FLUX.2 model/local-only/inventory contract changed")
    expected_isnet = {
        "model_path": str(ISNET_MODEL_PATH),
        "model_sha256": ISNET_MODEL_SHA256,
        "provenance_path": str(ISNET_PROVENANCE_PATH),
        "provenance_sha256": ISNET_PROVENANCE_SHA256,
        "inference_python": "/data/jzy/miniconda3/envs/hunyuan3d/bin/python",
        "python_version": "3.10.20",
        "rembg_version": "2.0.69",
        "onnxruntime_version": "1.23.2",
        "model_weight_license_status": "unresolved_no_license_sidecar_in_legacy_cache",
    }
    if job.get("isnet") != expected_isnet:
        raise ValueError("ISNet weight/code/environment/provenance contract changed")


def load_jobs(path: Path) -> list[dict[str, Any]]:
    path = _regular_file(path, "jobs JSON")
    payload, _ = route2_common.load_json_mapping_record(
        path,
        root=path.parent,
        description="jobs JSON",
        error_type=ValueError,
    )
    if not isinstance(payload, dict) or payload.get("schema") != JOBS_SCHEMA:
        raise ValueError(f"jobs schema must be {JOBS_SCHEMA}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("jobs must be a list")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        validate_job(job)
    if [job["case_id"] for job in jobs] != list(CASE_CONTRACTS):
        raise ValueError("jobs must preserve the exact case order")
    return jobs


def authenticate_source(job: Mapping[str, Any]) -> dict[str, Any]:
    validate_job(job)
    image = _regular_file(Path(job["source_image"]), "approved source image")
    review = _regular_file(Path(job["source_review"]), "approved source review")
    candidate_manifest_record = _validate_path_hash_record(
        job["source_candidate_manifest"], "source candidate manifest"
    )
    candidate_manifest = _regular_file(
        Path(candidate_manifest_record["path"]), "approved source candidate manifest"
    )
    if (
        image.name != "candidate.png"
        or review.name != "reference_review.json"
        or candidate_manifest.name != "candidate_manifest.json"
    ):
        raise ValueError("source must be the direct candidate/review/manifest triplet")
    if image.parent != review.parent or image.parent != candidate_manifest.parent:
        raise ValueError("source image, approval, and manifest must share one directory")
    image_file = route2_common.file_record(
        image,
        root=image.parent,
        description="approved source image",
        error_type=ValueError,
    )
    payload, review_file = route2_common.load_json_mapping_record(
        review,
        root=review.parent,
        description="approved source review",
        error_type=ValueError,
    )
    source_manifest, manifest_file = route2_common.load_json_mapping_record(
        candidate_manifest,
        root=candidate_manifest.parent,
        description="approved source candidate manifest",
        error_type=ValueError,
    )
    image_hash = image_file["sha256"]
    review_hash = review_file["sha256"]
    manifest_hash = manifest_file["sha256"]
    if (
        image_hash != job["source_image_sha256"]
        or review_hash != job["source_review_sha256"]
        or manifest_hash != candidate_manifest_record["sha256"]
    ):
        raise ValueError("approved source snapshot hash mismatch")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "human_reference_review_v1"
        or payload.get("asset_id") != job["base_asset_id"]
        or payload.get("decision") != "approved"
        or payload.get("candidate_sha256") != image_hash
        or payload.get("candidate_manifest_sha256") != manifest_hash
    ):
        raise ValueError("source reference is not the exact approved candidate")
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema_version") != "human_reference_candidate_v1"
        or source_manifest.get("asset_id") != job["base_asset_id"]
        or source_manifest.get("output_sha256") != image_hash
        or source_manifest.get("model_revision") != MODEL_REVISION
        or source_manifest.get("width") != WIDTH
        or source_manifest.get("height") != HEIGHT
        or source_manifest.get("steps") != STEPS
        or float(source_manifest.get("guidance_scale", -1)) != GUIDANCE_SCALE
    ):
        raise ValueError("approved source candidate provenance chain changed")
    with Image.open(image) as source:
        source.load()
        if source.size != (WIDTH, HEIGHT):
            raise ValueError("approved source canvas differs from the route-2 contract")
    alpha_record = _validate_path_hash_record(job["source_alpha"], "source alpha")
    rgba_record = _validate_path_hash_record(job["source_rgba"], "source RGBA")
    alpha = _regular_file(Path(alpha_record["path"]), "approved source alpha")
    rgba = _regular_file(Path(rgba_record["path"]), "approved source RGBA")
    alpha_file = route2_common.file_record(
        alpha,
        root=alpha.parent,
        description="approved source alpha",
        error_type=ValueError,
    )
    rgba_file = route2_common.file_record(
        rgba,
        root=rgba.parent,
        description="approved source RGBA",
        error_type=ValueError,
    )
    if alpha_file["sha256"] != alpha_record["sha256"] or rgba_file["sha256"] != rgba_record["sha256"]:
        raise ValueError("approved source alpha/RGBA snapshot hash mismatch")
    if alpha.name != "alpha_isnet.png" or rgba.name != "input_rgba_isnet.png" or alpha.parent != rgba.parent:
        raise ValueError("source alpha/RGBA paths are not canonical")
    with Image.open(image) as opened_source, Image.open(alpha) as opened_alpha, Image.open(
        rgba
    ) as opened_rgba:
        opened_source.load()
        opened_alpha.load()
        opened_rgba.load()
        if opened_alpha.mode != "L" or opened_rgba.mode != "RGBA":
            raise ValueError("approved source alpha/RGBA modes changed")
        if opened_alpha.size != (WIDTH, HEIGHT) or opened_rgba.size != (WIDTH, HEIGHT):
            raise ValueError("approved source alpha/RGBA canvas changed")
        if opened_alpha.tobytes() != opened_rgba.getchannel("A").tobytes():
            raise ValueError("approved source alpha and RGBA alpha channel differ")
        if opened_source.convert("RGB").tobytes() != opened_rgba.convert("RGB").tobytes():
            raise ValueError("approved source RGBA RGB channels differ from the approved image")
    return {
        "image": str(image),
        "image_sha256": image_hash,
        "image_size_bytes": image_file["size_bytes"],
        "review": str(review),
        "review_sha256": review_hash,
        "review_size_bytes": review_file["size_bytes"],
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "sha256": manifest_hash,
            "size_bytes": manifest_file["size_bytes"],
        },
        "source_alpha": {
            "path": str(alpha),
            "sha256": alpha_record["sha256"],
            "size_bytes": alpha_file["size_bytes"],
        },
        "source_rgba": {
            "path": str(rgba),
            "sha256": rgba_record["sha256"],
            "size_bytes": rgba_file["size_bytes"],
        },
        "source_rgba_rgb_matches_source": True,
    }


def authenticate_isnet(job: Mapping[str, Any]) -> dict[str, Any]:
    validate_job(job)
    model = _regular_file(ISNET_MODEL_PATH, "canonical ISNet model")
    model_file = route2_common.hash_file_snapshot(
        model,
        root=model.parent,
        description="canonical ISNet model",
        error_type=ValueError,
    )
    if model_file["sha256"] != ISNET_MODEL_SHA256 or model_file["size_bytes"] != 178648008:
        raise ValueError("canonical ISNet model hash or size changed")
    provenance = _regular_file(ISNET_PROVENANCE_PATH, "ISNet provenance")
    payload, provenance_file = route2_common.load_json_mapping_record(
        provenance,
        root=provenance.parent,
        description="ISNet provenance",
        error_type=ValueError,
    )
    if provenance_file["sha256"] != ISNET_PROVENANCE_SHA256:
        raise ValueError("ISNet provenance hash changed")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "isnet_local_weight_provenance_v1"
        or payload.get("canonical_weight", {}).get("path") != str(ISNET_MODEL_PATH)
        or payload.get("canonical_weight", {}).get("sha256") != ISNET_MODEL_SHA256
        or payload.get("inference_code", {}).get("version") != "2.0.69"
        or payload.get("environment", {}).get("onnxruntime_version") != "1.23.2"
        or payload.get("model_weight_license_status")
        != "unresolved_no_license_sidecar_in_legacy_cache"
        or payload.get("network_download_performed") is not False
    ):
        raise ValueError("ISNet provenance contract changed")
    license_path = _regular_file(
        Path(payload["inference_code"]["license_snapshot"]), "rembg license snapshot"
    )
    license_file = route2_common.file_record(
        license_path,
        root=license_path.parent,
        description="rembg license snapshot",
        error_type=ValueError,
    )
    if license_file["sha256"] != payload["inference_code"]["license_sha256"]:
        raise ValueError("rembg license snapshot hash changed")
    python = Path(payload["environment"]["python"]).absolute()
    environment_root = Path("/data/jzy/miniconda3/envs/hunyuan3d").resolve()
    try:
        resolved_python = python.resolve(strict=True)
        resolved_python.relative_to(environment_root)
    except (OSError, ValueError) as error:
        raise ValueError("pinned ISNet inference Python resolves outside its environment") from error
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise ValueError("pinned ISNet inference Python is unavailable")
    return {
        "model": {
            "path": str(model),
            "sha256": ISNET_MODEL_SHA256,
            "size_bytes": model_file["size_bytes"],
        },
        "provenance": {
            "path": str(provenance),
            "sha256": ISNET_PROVENANCE_SHA256,
            "size_bytes": provenance_file["size_bytes"],
        },
        "code_license": {
            "path": str(license_path),
            "sha256": license_file["sha256"],
            "size_bytes": license_file["size_bytes"],
            "spdx": "MIT",
        },
        "model_weight_license_status": payload["model_weight_license_status"],
        "environment": dict(payload["environment"]),
    }


def authenticate_mask_bundle(
    job: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    record = _validate_path_hash_record(job["mask_bundle"], "mask bundle")
    manifest_path = _regular_file(Path(record["path"]), "mask bundle manifest")
    manifest, manifest_file = route2_common.load_json_mapping_record(
        manifest_path,
        root=manifest_path.parent,
        description="mask bundle manifest",
        error_type=ValueError,
        require_mode=0o444,
    )
    if manifest_path.name != "mask_manifest.json" or manifest_file["sha256"] != record["sha256"]:
        raise ValueError("mask bundle manifest path or hash changed")
    case_id = job["case_id"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "human_attribute_mask_bundle_v2"
        or manifest.get("case_id") != case_id
        or manifest.get("base_asset_id") != job["base_asset_id"]
        or manifest.get("construction_version") != semantic_masks.MASK_CONSTRUCTION_VERSION
        or manifest.get("strategy") != CASE_CONTRACTS[case_id]["strategy"]
        or manifest.get("target_parameters") != job["target_parameters"]
        or manifest.get("quantitative_gates")
        != CASE_CONTRACTS[case_id]["quantitative_gates"]
        or manifest.get("agent_visual_review") != "pending_agent_mask_overlay_qa"
        or manifest.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(manifest)
    ):
        raise ValueError("mask bundle contract or review state changed")
    if (
        manifest.get("source_image", {}).get("sha256") != source["image_sha256"]
        or manifest.get("source_alpha", {}).get("sha256")
        != source["source_alpha"]["sha256"]
    ):
        raise ValueError("mask bundle does not bind the authenticated source")
    expected_names = {
        "edit_core.png",
        "transition_band.png",
        "protected_guard.png",
        "overlay.png",
    }
    artifacts = manifest.get("assets")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise ValueError("mask bundle artifact set is incomplete")
    authenticated: dict[str, Any] = {}
    images: dict[str, Image.Image] = {}
    for filename in sorted(expected_names):
        path = _regular_file(manifest_path.parent / filename, f"mask artifact {filename}")
        descriptor = artifacts[filename]
        artifact_file = route2_common.file_record(
            path,
            root=manifest_path.parent,
            description=f"mask artifact {filename}",
            error_type=ValueError,
            require_mode=0o444,
        )
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("path") != str(path)
            or descriptor.get("sha256") != artifact_file["sha256"]
            or descriptor.get("size_bytes") != artifact_file["size_bytes"]
        ):
            raise ValueError(f"mask artifact descriptor changed: {filename}")
        with Image.open(path) as opened:
            opened.load()
            if opened.size != (WIDTH, HEIGHT):
                raise ValueError(f"mask artifact canvas changed: {filename}")
            images[filename] = opened.copy()
        authenticated[filename] = {
            "path": str(path),
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
        }
    core = np.asarray(images["edit_core.png"].convert("L"), dtype=np.uint8)
    band = np.asarray(images["transition_band.png"].convert("L"), dtype=np.uint8)
    guard = np.asarray(images["protected_guard.png"].convert("L"), dtype=np.uint8)
    if any(set(np.unique(values).tolist()) != {0, 255} for values in (core, band, guard)):
        raise ValueError("three-layer mask bundle is not nonempty binary")
    partition = (core > 0).astype(np.uint8) + (band > 0) + (guard > 0)
    if not np.all(partition == 1):
        raise ValueError("three-layer masks do not form an exact partition")
    with Image.open(source["source_alpha"]["path"]) as opened_alpha:
        source_alpha_bytes_hash = hashlib.sha256(opened_alpha.convert("L").tobytes()).hexdigest()
    if manifest.get("metrics", {}).get("source_alpha_sha256") != source_alpha_bytes_hash:
        raise ValueError("mask bundle decoded source alpha hash changed")
    decision_record = _validate_path_hash_record(job["mask_decision"], "mask decision")
    expected_decision = semantic_masks.mask_agent_decision_path(manifest_path.parent)
    decision_path = _regular_file(Path(decision_record["path"]), "mask agent QA decision")
    decision_file = route2_common.file_record(
        decision_path,
        root=decision_path.parent,
        description="mask agent QA decision",
        error_type=ValueError,
        require_mode=0o444,
    )
    if (
        decision_path != expected_decision
        or decision_file["sha256"] != decision_record["sha256"]
    ):
        raise ValueError("mask agent QA decision path or hash changed")
    try:
        decision = semantic_masks.assert_mask_agent_qa_passed(manifest_path.parent)
    except ValueError as error:
        raise ValueError(f"mask bundle agent QA has not passed: {error}") from error
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": record["sha256"],
            "size_bytes": manifest_file["size_bytes"],
        },
        "strategy": manifest["strategy"],
        "target_parameters": manifest["target_parameters"],
        "quantitative_gates": manifest["quantitative_gates"],
        "metrics": manifest["metrics"],
        "assets": authenticated,
        "partition_exact": True,
        "agent_decision": {
            "path": str(decision_path),
            "sha256": decision_record["sha256"],
            "size_bytes": decision_file["size_bytes"],
            "status": decision["status"],
        },
    }


def _validate_preflight_output_root(output_root: Path) -> Path:
    output_root = Path(output_root).absolute()
    existing = output_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if existing.is_symlink() or not existing.is_dir() or existing.resolve() != existing:
        raise ValueError("attribute output path has no direct real ancestor")
    if output_root.exists() and (
        output_root.is_symlink()
        or not output_root.is_dir()
        or output_root.resolve() != output_root
    ):
        raise ValueError("attribute output root must be a direct real directory")
    return output_root


def authenticate_base_route2_qualification(job: Mapping[str, Any]) -> dict[str, Any]:
    validate_job(job)
    pointer = Path(job["base_qualified_candidate"]).absolute()
    try:
        pointer_record = route2_common.file_record(
            pointer,
            root=pointer.parent,
            description="base qualified candidate",
            error_type=ValueError,
            require_mode=0o444,
        )
        qualified = qualified_candidate.validate_qualified_candidate(pointer)
    except (ValueError, qualified_candidate.QualificationError) as error:
        raise ValueError(
            f"base Route-2 qualification is absent or stale for {job['base_asset_id']}: {error}"
        ) from error
    final_branch = qualified.get("final_branch")
    dynamic = qualified.get("dynamic")
    if (
        qualified.get("asset_id") != job["base_asset_id"]
        or qualified.get("base_avatar_id") != job["base_asset_id"]
        or qualified.get("status")
        != "agent_qa_passed_pending_user_acceptance"
        or not isinstance(final_branch, Mapping)
        or set(final_branch) != {"branch_id", "path", "relative_root"}
        or not isinstance(dynamic, Mapping)
        or not isinstance(dynamic.get("review_dir"), str)
    ):
        raise ValueError("base Route-2 qualification asset lineage changed")
    return {
        "asset_id": job["base_asset_id"],
        "status": qualified["status"],
        "qualified_candidate": {
            "path": str(pointer),
            "sha256": pointer_record["sha256"],
            "size_bytes": pointer_record["size_bytes"],
        },
        "final_branch": dict(final_branch),
        "review_dir": dynamic["review_dir"],
    }


def preflight_batch(
    jobs_path: Path,
    output_root: Path,
    *,
    authenticate_model: bool = True,
    require_base_qa: bool = True,
) -> dict[str, Any]:
    """Authenticate the whole batch without importing torch or touching a GPU."""
    jobs_path = _regular_file(jobs_path, "jobs JSON")
    jobs = load_jobs(jobs_path)
    output_root = _validate_preflight_output_root(output_root)
    descriptor = {
        "path": str(jobs_path),
        "sha256": sha256_file(jobs_path),
        "size_bytes": jobs_path.stat().st_size,
    }
    model = None
    if authenticate_model:
        model = authenticate_model_snapshot(
            model_root=MODEL_ROOT,
            revision=MODEL_REVISION,
            inventory_path=MODEL_INVENTORY_PATH,
            inventory_sha256=MODEL_INVENTORY_SHA256,
            expected_model_name="black-forest-labs/FLUX.2-klein-4B",
        )
    records = []
    for job in jobs:
        source = authenticate_source(job)
        mask_bundle = authenticate_mask_bundle(job, source)
        isnet = authenticate_isnet(job)
        base_qualification = (
            authenticate_base_route2_qualification(job)
            if require_base_qa
            else {
                "asset_id": job["base_asset_id"],
                "qualified_candidate": {
                    "path": job["base_qualified_candidate"],
                },
                "status": "not_checked_configuration_only",
            }
        )
        destination = output_root / job["case_id"]
        existing = _existing_success(destination, job["case_id"])
        records.append(
            {
                "job": dict(job),
                "job_sha256": hashlib.sha256(
                    json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "source": source,
                "mask_bundle": mask_bundle,
                "isnet": isnet,
                "base_qualification": base_qualification,
                "destination": str(destination),
                "resume_status": "existing_success" if existing else "ready",
            }
        )
    return {
        "schema": "flux2_human_attribute_preflight_v2",
        "jobs_path": str(jobs_path),
        "jobs_sha256": descriptor["sha256"],
        "jobs_descriptor": descriptor,
        "runner": {
            "path": str(RUNNER_PATH),
            "sha256": sha256_file(RUNNER_PATH),
            "size_bytes": RUNNER_PATH.stat().st_size,
        },
        "model": model,
        "output_root": str(output_root),
        "jobs": records,
        "gpu_touched": False,
        "base_qa_required": require_base_qa,
        "execution_authorized": bool(
            authenticate_model
            and model is not None
            and require_base_qa
            and all(
                record["base_qualification"].get("status")
                == "agent_qa_passed_pending_user_acceptance"
                for record in records
            )
        ),
    }


def _artifact_record(path: Path, public_path: Path) -> dict[str, Any]:
    return {
        "path": str(public_path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_candidate_bundle(
    *,
    staging: Path,
    public_destination: Path,
    preflight_record: Mapping[str, Any],
    jobs_descriptor: Mapping[str, Any],
    raw_candidate: Image.Image,
    predicted_alpha: Image.Image | None,
) -> Path:
    """Build one complete reviewable bundle without publishing its directory."""
    staging = Path(staging).absolute()
    public_destination = Path(public_destination).absolute()
    if staging.is_symlink() or not staging.is_dir() or any(staging.iterdir()):
        raise ValueError("candidate staging must be an empty direct directory")
    job = preflight_record.get("job")
    source_record = preflight_record.get("source")
    mask_record = preflight_record.get("mask_bundle")
    base_qualification = preflight_record.get("base_qualification")
    if not isinstance(job, Mapping) or not isinstance(source_record, Mapping) or not isinstance(mask_record, Mapping):
        raise ValueError("candidate bundle requires a complete preflight record")
    validate_job(job)
    if (
        not isinstance(base_qualification, Mapping)
        or base_qualification.get("asset_id") != job["base_asset_id"]
        or base_qualification.get("qualified_candidate", {}).get("path")
        != job["base_qualified_candidate"]
        or base_qualification.get("status")
        != "agent_qa_passed_pending_user_acceptance"
    ):
        raise ValueError("candidate bundle requires a qualified Route-2 base asset")
    case_id = str(job["case_id"])
    with Image.open(source_record["image"]) as opened:
        source = opened.convert("RGB")
    raw = raw_candidate.convert("RGB")
    if raw.size != source.size:
        raise ValueError("FLUX.2 raw candidate canvas changed")
    mask_images: dict[str, Image.Image] = {}
    for filename in ("edit_core.png", "transition_band.png", "protected_guard.png"):
        path = Path(mask_record["assets"][filename]["path"])
        with Image.open(path) as opened:
            mask_images[filename] = opened.convert("L")
    candidate, pixel_proof = semantic_masks.feathered_composite(
        source,
        raw,
        mask_images["edit_core.png"],
        mask_images["transition_band.png"],
    )
    gates = mask_record["quantitative_gates"]
    if (
        pixel_proof["outside_changed_pixels"]
        > gates["outside_band_changed_pixels_max"]
        or pixel_proof["outside_max_abs_channel_delta"]
        > gates["outside_band_max_abs_channel_delta"]
        or not pixel_proof["transition_is_feathered"]
        or pixel_proof["inside_changed_pixels"] == 0
    ):
        raise ValueError("candidate failed exact outside-band or feathered-transition gates")
    with Image.open(source_record["source_alpha"]["path"]) as opened:
        source_alpha = opened.convert("L")
    candidate_alpha, alpha_proof = semantic_masks.build_candidate_alpha(
        case_id,
        source_alpha,
        predicted_alpha,
        mask_images["edit_core.png"],
        mask_images["transition_band.png"],
    )
    if alpha_proof["outside_changed_pixels"] != 0:
        raise ValueError("candidate alpha changed outside the authorized mask bundle")
    case_metrics = semantic_masks.evaluate_candidate_metrics(
        case_id,
        source=source,
        candidate=candidate,
        source_alpha=source_alpha,
        candidate_alpha=candidate_alpha,
        edit_core=mask_images["edit_core.png"],
        transition_band=mask_images["transition_band.png"],
    )
    if case_metrics.get("passed") is not True:
        failed = sorted(
            name
            for name, passed in case_metrics.get("checks", {}).items()
            if passed is not True
        )
        raise ValueError(
            "candidate failed case-specific quantitative gates: "
            + ", ".join(failed)
        )
    rgba = candidate.convert("RGBA")
    rgba.putalpha(candidate_alpha)
    overlay = semantic_masks.render_mask_overlay(
        candidate,
        mask_images["edit_core.png"],
        mask_images["transition_band.png"],
    )
    difference = semantic_masks.render_difference(source, candidate)
    images = {
        "source.png": source,
        "raw_candidate.png": raw,
        "candidate.png": candidate,
        "source_alpha.png": source_alpha,
        "candidate_alpha.png": candidate_alpha,
        "candidate_rgba.png": rgba,
        "edit_core.png": mask_images["edit_core.png"],
        "transition_band.png": mask_images["transition_band.png"],
        "protected_guard.png": mask_images["protected_guard.png"],
        "overlay.png": overlay,
        "diff.png": difference,
    }
    for filename, image in images.items():
        image.save(staging / filename, format="PNG")
    image_records = {
        filename: _artifact_record(staging / filename, public_destination / filename)
        for filename in sorted(images)
    }
    quantitative_snapshot = {
        "mask_gates": dict(gates),
        "pixel_proof": pixel_proof,
        "alpha_proof": alpha_proof,
        "case_metrics": case_metrics,
        "mask_metrics": dict(mask_record["metrics"]),
        "automatic_checks": "passed",
    }
    pending_decision = {
        "schema": "human_attribute_agent_2d_decision_v1",
        "case_id": case_id,
        "status": "pending_agent_2d_visual_qa",
        "reviewer_kind": "agent",
        "quantitative_snapshot": quantitative_snapshot,
        "review_required": {
            "full_resolution_source": image_records["source.png"],
            "full_resolution_candidate": image_records["candidate.png"],
            "mask_overlay": image_records["overlay.png"],
            "difference": image_records["diff.png"],
            "candidate_rgba": image_records["candidate_rgba.png"],
        },
        "user_acceptance": "pending_user_review",
    }
    decision_path = staging / "agent_2d_decision.json"
    decision_path.write_bytes(_json_bytes(pending_decision))
    decision_path.chmod(0o444)
    artifacts = {
        **image_records,
        "agent_2d_decision.json": _artifact_record(
            decision_path, public_destination / "agent_2d_decision.json"
        ),
    }
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "case_id": case_id,
        "base_asset_id": job["base_asset_id"],
        "downstream_asset_id": job["downstream_asset_id"],
        "state_classification": "research_candidate",
        "bundle_status": "generated_pending_agent_2d_visual_qa",
        "agent_qa_status": "pending_agent_2d_visual_qa",
        "user_acceptance": "pending_user_review",
        "created_at_utc": _utc_now(),
        "source": dict(source_record),
        "base_route2_qualification": dict(base_qualification),
        "jobs": dict(jobs_descriptor),
        "job_sha256": preflight_record["job_sha256"],
        "runner": {
            "path": str(RUNNER_PATH),
            "sha256": sha256_file(RUNNER_PATH),
            "size_bytes": RUNNER_PATH.stat().st_size,
        },
        "model": dict(job["model"]),
        "isnet": dict(preflight_record["isnet"]),
        "parameters": {
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "seed": job["seed"],
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "physical_gpu": PHYSICAL_GPU,
        },
        "mask_bundle": dict(mask_record),
        "target_parameters": dict(job["target_parameters"]),
        "quantitative_snapshot": quantitative_snapshot,
        "artifacts": artifacts,
    }
    encoded = json.dumps(manifest, sort_keys=True)
    if "user_approved" in encoded:
        raise ValueError("candidate manifest may not claim user approval")
    manifest_path = staging / "candidate_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    for path in staging.iterdir():
        path.chmod(0o444)
    return manifest_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    result = function(
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(destination)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), destination)


def _atomic_readonly_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path).absolute()
    if path.parent.is_symlink() or not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise ValueError(f"JSON publication parent must be a direct real directory: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
        return path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _replace_readonly_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    identity_field: str | None = "attempt_id",
) -> Path:
    """Durably replace one already-created attempt ledger with the same identity."""
    path = _regular_file(path, "attempt ledger")
    existing, _ = route2_common.load_json_mapping_record(
        path,
        root=path.parent,
        description="attempt ledger",
        error_type=ValueError,
        require_mode=0o444,
    )
    if identity_field is not None and existing.get(identity_field) != payload.get(
        identity_field
    ):
        raise ValueError("attempt ledger identity changed")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return path
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_tree(path: Path) -> None:
    root = Path(path).absolute()
    directories = [root]
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"attempt staging contains a symlink: {item}")
        if item.is_dir():
            directories.append(item)
            continue
        descriptor = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _embed_success_attempt(
    *,
    staging: Path,
    public_destination: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Put the final immutable attempt proof in the still-unpublished bundle."""
    manifest_path = Path(staging) / "candidate_manifest.json"
    manifest, _ = route2_common.load_json_mapping_record(
        manifest_path,
        root=staging,
        description="staged attribute candidate manifest",
        error_type=ValueError,
    )
    if (
        manifest.get("schema") != OUTPUT_SCHEMA
        or manifest.get("case_id") != ledger.get("case_id")
        or not isinstance(manifest.get("artifacts"), Mapping)
        or "generation_attempt.json" in manifest["artifacts"]
    ):
        raise ValueError("staged candidate cannot bind the final attempt ledger")
    attempt_path = Path(staging) / "generation_attempt.json"
    _atomic_readonly_json(attempt_path, ledger)
    descriptor = _artifact_record(
        attempt_path, Path(public_destination) / "generation_attempt.json"
    )
    manifest = dict(manifest)
    manifest["artifacts"] = {
        **dict(manifest["artifacts"]),
        "generation_attempt.json": descriptor,
    }
    manifest["generation_attempt"] = descriptor
    manifest_path.chmod(0o444)
    _replace_readonly_json(
        manifest_path,
        {**manifest, "attempt_id": ledger["attempt_id"]},
        identity_field=None,
    )
    return descriptor


def _make_readonly_tree(path: Path) -> None:
    for item in sorted(Path(path).rglob("*"), reverse=True):
        if item.is_file() and not item.is_symlink():
            item.chmod(0o444)


def _existing_success(destination: Path, case_id: str) -> Path | None:
    if not os.path.lexists(destination):
        return None
    if destination.is_symlink() or not destination.is_dir() or destination.resolve() != destination:
        raise ValueError(f"existing candidate output is not a direct directory: {destination}")
    manifest = destination / "candidate_manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.resolve() != manifest:
        raise ValueError(f"existing candidate has no authenticated manifest: {destination}")
    try:
        snapshot = attribute_review.validated_candidate_snapshot(destination)
    except attribute_review.AttributeReviewError as error:
        raise ValueError(f"existing output is not a full authenticated candidate snapshot: {error}") from error
    if (
        snapshot.get("case_id") != case_id
        or snapshot.get("candidate_manifest_path") != str(manifest)
    ):
        raise ValueError("existing authenticated candidate snapshot has changed lineage")
    return manifest


def execute_with_attempt(
    *,
    case_id: str,
    output_root: Path,
    job_descriptor: Mapping[str, Any],
    operation: Callable[[Path], Any],
) -> dict[str, Any]:
    """Execute one operation with append-only attempts and preserved failures."""
    if case_id not in CASE_CONTRACTS:
        raise ValueError(f"unknown attribute case: {case_id}")
    output_root = Path(output_root).absolute()
    if output_root.is_symlink() or not output_root.is_dir() or output_root.resolve() != output_root:
        raise ValueError("attribute output root must be a direct real directory")
    destination = output_root / case_id
    existing = _existing_success(destination, case_id)
    if existing is not None:
        return {
            "case_id": case_id,
            "status": "existing_success",
            "manifest": str(existing),
        }
    attempt_id = uuid.uuid4().hex
    attempts_root = output_root / ".attempts" / case_id
    failures_root = output_root / ".failed_attempts" / case_id
    attempts_root.mkdir(parents=True, exist_ok=True)
    failures_root.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in (attempts_root, failures_root)):
        raise ValueError("attempt evidence roots must not be symlinks")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{case_id}.{attempt_id}.", suffix=".staging", dir=output_root
        )
    )
    ledger_path = attempts_root / f"{attempt_id}.json"
    started_at = _utc_now()
    started_ledger = {
        "schema": "flux2_human_attribute_attempt_v2",
        "attempt_id": attempt_id,
        "case_id": case_id,
        "status": "started",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "job": dict(job_descriptor),
        "staging": {
            "path": str(staging),
            "created": True,
            "preserved_after_success": False,
        },
        "publication": {
            "policy": "atomic_no_replace",
            "destination": str(destination),
            "staged_before_publication": False,
        },
        "error": None,
    }
    _atomic_readonly_json(ledger_path, started_ledger)
    try:
        operation_result = operation(staging)
        staged_manifest = staging / "candidate_manifest.json"
        if (
            staged_manifest.is_symlink()
            or not staged_manifest.is_file()
            or staged_manifest.resolve() != staged_manifest
        ):
            raise ValueError("successful operation did not create a direct candidate_manifest.json")
        ledger = {
            "schema": "flux2_human_attribute_attempt_v2",
            "attempt_id": attempt_id,
            "case_id": case_id,
            "status": "generated",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "job": dict(job_descriptor),
            "staging": {
                "path": str(staging),
                "created": True,
                "preserved_after_success": False,
            },
            "publication": {
                "policy": "atomic_no_replace",
                "destination": str(destination),
                "staged_before_publication": True,
            },
            "preflight_reauthenticated": bool(
                isinstance(operation_result, Mapping)
                and operation_result.get("preflight_reauthenticated") is True
            ),
            "postflight_reauthenticated": bool(
                isinstance(operation_result, Mapping)
                and operation_result.get("postflight_reauthenticated") is True
            ),
            "preflight_snapshot_sha256": (
                operation_result.get("preflight_snapshot_sha256")
                if isinstance(operation_result, Mapping)
                else None
            ),
            "postflight_snapshot_sha256": (
                operation_result.get("postflight_snapshot_sha256")
                if isinstance(operation_result, Mapping)
                else None
            ),
            "error": None,
        }
        _embed_success_attempt(
            staging=staging,
            public_destination=destination,
            ledger=ledger,
        )
        _replace_readonly_json(ledger_path, ledger)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        _make_readonly_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, destination)
        _fsync_directory(output_root)
        manifest = destination / "candidate_manifest.json"
        return {
            "case_id": case_id,
            "status": "generated",
            "attempt_id": attempt_id,
            "manifest": str(manifest),
            "ledger": str(ledger_path),
            "operation_result": operation_result,
        }
    except BaseException as error:
        evidence = failures_root / attempt_id
        if staging.exists():
            _make_readonly_tree(staging)
            _fsync_tree(staging)
            _rename_noreplace(staging, evidence)
            _fsync_directory(failures_root)
        ledger = {
            "schema": "flux2_human_attribute_attempt_v2",
            "attempt_id": attempt_id,
            "case_id": case_id,
            "status": "rejected_generation_failure",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "job": dict(job_descriptor),
            "evidence_dir": str(evidence),
            "staging": started_ledger["staging"],
            "publication": {
                "policy": "atomic_no_replace",
                "destination": str(destination),
                "staged_before_publication": False,
            },
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _replace_readonly_json(ledger_path, ledger)
        if not isinstance(error, Exception):
            raise
        return {
            "case_id": case_id,
            "status": "rejected_generation_failure",
            "attempt_id": attempt_id,
            "ledger": str(ledger_path),
            "evidence_dir": str(evidence),
            "error": ledger["error"],
        }


def authenticate_model_snapshot(
    *,
    model_root: Path,
    revision: str,
    inventory_path: Path,
    inventory_sha256: str,
    expected_model_name: str,
) -> dict[str, Any]:
    """Authenticate every exact snapshot file, license, and incomplete state."""
    model_root = Path(model_root).absolute()
    inventory_path = _regular_file(inventory_path, "model snapshot inventory")
    inventory, inventory_file = route2_common.load_json_mapping_record(
        inventory_path,
        root=inventory_path.parent,
        description="model snapshot inventory",
        error_type=ValueError,
    )
    if inventory_file["sha256"] != inventory_sha256:
        raise ValueError("model snapshot inventory SHA-256 changed")
    if model_root.is_symlink() or not model_root.is_dir() or model_root.resolve() != model_root:
        raise ValueError("model cache root must be a direct real directory")
    incomplete = sorted(
        path for path in model_root.rglob("*") if path.name.endswith(".incomplete")
    )
    if incomplete:
        raise ValueError(f"model cache contains incomplete file: {incomplete[0]}")
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != "huggingface_snapshot_inventory_v1"
        or inventory.get("model") != expected_model_name
        or inventory.get("revision") != revision
        or not isinstance(inventory.get("files"), list)
    ):
        raise ValueError("model inventory schema, model, or revision changed")
    snapshot = model_root / "snapshots" / revision
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.resolve() != snapshot:
        raise ValueError("pinned model snapshot must be a direct real directory")
    records: dict[str, Mapping[str, Any]] = {}
    for record in inventory["files"]:
        if not isinstance(record, Mapping):
            raise ValueError("model inventory file record is invalid")
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in records
        ):
            raise ValueError("model inventory relative path is invalid or duplicated")
        _validate_hash(record.get("sha256"), f"model file {relative} SHA-256")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ValueError(f"model file {relative} size is invalid")
        records[relative] = record
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if actual != set(records):
        raise ValueError("model snapshot exact file set differs from inventory")
    authenticated: dict[str, Any] = {}
    for relative, record in sorted(records.items()):
        logical = snapshot / relative
        resolved = logical.resolve()
        try:
            resolved.relative_to(model_root)
        except ValueError as error:
            raise ValueError(f"model file resolves outside cache root: {relative}") from error
        if not resolved.is_file() or resolved.stat().st_size != record["size_bytes"]:
            raise ValueError(f"model file size changed: {relative}")
        if sha256_file(resolved) != record["sha256"]:
            raise ValueError(f"model file SHA-256 changed: {relative}")
        authenticated[relative] = {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    license_relative = inventory.get("license_relative_path")
    if license_relative not in authenticated:
        raise ValueError("model inventory license snapshot is missing")
    return {
        "name": expected_model_name,
        "root": str(model_root),
        "revision": revision,
        "snapshot": str(snapshot),
        "inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha256,
            "size_bytes": inventory_file["size_bytes"],
        },
        "file_count": len(authenticated),
        "files": authenticated,
        "license": authenticated[license_relative],
        "local_files_only": True,
    }


def _assert_execution_preflight(preflight: Mapping[str, Any]) -> None:
    if not isinstance(preflight, Mapping) or preflight.get("execution_authorized") is not True:
        raise ValueError("operation requires an execution-authorized preflight")
    model = preflight.get("model")
    records = preflight.get("jobs")
    if (
        preflight.get("schema") != "flux2_human_attribute_preflight_v2"
        or preflight.get("gpu_touched") is not False
        or preflight.get("base_qa_required") is not True
        or preflight.get("runner", {}).get("sha256") != sha256_file(RUNNER_PATH)
        or not isinstance(model, Mapping)
        or model.get("name") != "black-forest-labs/FLUX.2-klein-4B"
        or model.get("revision") != MODEL_REVISION
        or model.get("root") != str(MODEL_ROOT)
        or model.get("snapshot")
        != str(MODEL_ROOT / "snapshots" / MODEL_REVISION)
        or not isinstance(model.get("inventory"), Mapping)
        or model.get("inventory", {}).get("path") != str(MODEL_INVENTORY_PATH)
        or model.get("inventory", {}).get("sha256") != MODEL_INVENTORY_SHA256
        or model.get("local_files_only") is not True
        or not isinstance(records, list)
        or any(not isinstance(record, Mapping) for record in records)
        or [record.get("job", {}).get("case_id") for record in records]
        != list(CASE_CONTRACTS)
    ):
        raise ValueError("execution-authorized preflight is incomplete or stale")
    for record in records:
        job = record.get("job")
        qualification = record.get("base_qualification")
        if not isinstance(job, Mapping) or not isinstance(qualification, Mapping):
            raise ValueError("execution-authorized job record is incomplete")
        validate_job(job)
        if (
            qualification.get("asset_id") != job["base_asset_id"]
            or qualification.get("qualified_candidate", {}).get("path")
            != job["base_qualified_candidate"]
            or qualification.get("status")
            != "agent_qa_passed_pending_user_acceptance"
        ):
            raise ValueError("execution-authorized base qualification changed")


def _execution_snapshot_expected(
    preflight: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "jobs": dict(preflight["jobs_descriptor"]),
        "runner": dict(preflight["runner"]),
        "model": dict(preflight["model"]),
        "job": dict(record["job"]),
        "job_sha256": record["job_sha256"],
        "source": dict(record["source"]),
        "mask_bundle": dict(record["mask_bundle"]),
        "isnet": dict(record["isnet"]),
        "base_qualification": dict(record["base_qualification"]),
    }


def _current_execution_input_snapshot(
    preflight: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    _assert_execution_preflight(preflight)
    job = record["job"]
    jobs_path = Path(preflight["jobs_path"]).absolute()
    jobs_file = route2_common.file_record(
        jobs_path,
        root=jobs_path.parent,
        description="execution jobs contract",
        error_type=ValueError,
        require_mode=0o444,
    )
    runner_file = route2_common.file_record(
        RUNNER_PATH,
        root=RUNNER_PATH.parent,
        description="FLUX attribute runner",
        error_type=ValueError,
    )
    source = authenticate_source(job)
    return {
        "jobs": {
            "path": jobs_file["path"],
            "sha256": jobs_file["sha256"],
            "size_bytes": jobs_file["size_bytes"],
        },
        "runner": {
            "path": runner_file["path"],
            "sha256": runner_file["sha256"],
            "size_bytes": runner_file["size_bytes"],
        },
        "model": authenticate_model_snapshot(
            model_root=MODEL_ROOT,
            revision=MODEL_REVISION,
            inventory_path=MODEL_INVENTORY_PATH,
            inventory_sha256=MODEL_INVENTORY_SHA256,
            expected_model_name="black-forest-labs/FLUX.2-klein-4B",
        ),
        "job": dict(job),
        "job_sha256": hashlib.sha256(
            json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source": source,
        "mask_bundle": authenticate_mask_bundle(job, source),
        "isnet": authenticate_isnet(job),
        "base_qualification": authenticate_base_route2_qualification(job),
    }


def reauthenticate_execution_inputs(
    preflight: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Twice authenticate all inference inputs and compare to the frozen preflight."""
    _assert_execution_preflight(preflight)
    expected = _execution_snapshot_expected(preflight, record)
    current = route2_common.stable_mapping_snapshot(
        lambda: _current_execution_input_snapshot(preflight, record),
        ValueError,
        f"FLUX inference inputs for {record.get('job', {}).get('case_id', 'unknown')}",
    )
    if route2_common.canonical_json(current) != route2_common.canonical_json(expected):
        raise ValueError("FLUX inference inputs changed after execution preflight")
    return current


def run_authenticated_inference(
    preflight: Mapping[str, Any],
    record: Mapping[str, Any],
    pipeline: Any,
    *,
    alpha_predictor: Callable[[Image.Image, Mapping[str, Any]], Image.Image] | None,
) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    """Bracket all image inference with exact twice-stable reauthentication."""
    before = reauthenticate_execution_inputs(preflight, record)
    raw = _run_flux_inference(record, pipeline)
    predicted_alpha = None
    if record["job"]["case_id"] not in {
        "short_sleeve_color",
        "trousers",
        "shoes",
    }:
        preview = _precompose_for_alpha(record, raw)
        predicted_alpha = alpha_predictor(preview, record["isnet"])
    after = reauthenticate_execution_inputs(preflight, record)
    return raw, predicted_alpha, {
        "preflight_reauthenticated": True,
        "postflight_reauthenticated": True,
        "preflight_snapshot": before,
        "postflight_snapshot": after,
    }


def load_pipeline(preflight: Mapping[str, Any]):
    _assert_execution_preflight(preflight)
    if (
        preflight.get("schema") != "flux2_human_attribute_preflight_v2"
        or preflight.get("gpu_touched") is not False
        or not isinstance(preflight.get("model"), Mapping)
        or preflight["model"].get("revision") != MODEL_REVISION
        or preflight.get("runner", {}).get("sha256") != sha256_file(RUNNER_PATH)
    ):
        raise ValueError("FLUX pipeline load requires a complete current preflight")
    snapshot = Path(preflight["model"]["snapshot"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", PHYSICAL_GPU):
        raise ValueError(f"CUDA_VISIBLE_DEVICES must be unset or {PHYSICAL_GPU}, got {visible!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    )
    return pipeline.to("cuda")


def run_preflighted_jobs(
    preflight: Mapping[str, Any],
    output_root: Path,
    pipeline: Any,
    *,
    alpha_predictor: Callable[[Image.Image, Mapping[str, Any]], Image.Image] | None = None,
) -> list[dict[str, Any]]:
    """Execute exactly the immutable records already authenticated by preflight."""
    _assert_execution_preflight(preflight)
    if (
        preflight.get("schema") != "flux2_human_attribute_preflight_v2"
        or preflight.get("runner", {}).get("sha256") != sha256_file(RUNNER_PATH)
        or preflight.get("gpu_touched") is not False
    ):
        raise ValueError("batch execution requires a current complete preflight")
    output_root = Path(output_root).absolute()
    if output_root != Path(preflight["output_root"]):
        raise ValueError("execution output root differs from preflight")
    if not output_root.exists():
        output_root.mkdir(parents=True)
    if output_root.is_symlink() or not output_root.is_dir() or output_root.resolve() != output_root:
        raise ValueError("execution output root must be a direct real directory")
    predictor = default_isnet_alpha_predictor if alpha_predictor is None else alpha_predictor
    results = []
    for record in preflight["jobs"]:
        job = record["job"]
        case_id = job["case_id"]

        def operation(staging: Path) -> dict[str, Any]:
            raw, predicted_alpha, reauthentication = run_authenticated_inference(
                preflight,
                record,
                pipeline,
                alpha_predictor=predictor,
            )
            manifest = build_candidate_bundle(
                staging=staging,
                public_destination=output_root / case_id,
                preflight_record=record,
                jobs_descriptor=preflight["jobs_descriptor"],
                raw_candidate=raw,
                predicted_alpha=predicted_alpha,
            )
            return {
                "manifest": str(manifest),
                "preflight_reauthenticated": reauthentication[
                    "preflight_reauthenticated"
                ],
                "postflight_reauthenticated": reauthentication[
                    "postflight_reauthenticated"
                ],
                "preflight_snapshot_sha256": hashlib.sha256(
                    route2_common.canonical_json(
                        reauthentication["preflight_snapshot"]
                    ).encode()
                ).hexdigest(),
                "postflight_snapshot_sha256": hashlib.sha256(
                    route2_common.canonical_json(
                        reauthentication["postflight_snapshot"]
                    ).encode()
                ).hexdigest(),
            }

        results.append(
            execute_with_attempt(
                case_id=case_id,
                output_root=output_root,
                job_descriptor={
                    "batch": dict(preflight["jobs_descriptor"]),
                    "job_sha256": record["job_sha256"],
                    "base_asset_id": job["base_asset_id"],
                    "downstream_asset_id": job["downstream_asset_id"],
                },
                operation=operation,
            )
        )
    return results


def _run_flux_inference(record: Mapping[str, Any], pipeline: Any) -> Image.Image:
    import torch

    job = record["job"]
    with Image.open(record["source"]["image"]) as opened:
        source = opened.convert("RGB")
    generator = torch.Generator("cuda").manual_seed(int(job["seed"]))
    inference_prompt = (
        f"{job['prompt']} Preserve the same identity, soft T-pose, front camera, "
        "limb gaps, lighting, and every protected non-target attribute. "
        f"Avoid: {job['negative_prompt']}."
    )
    result = pipeline(
        image=source,
        prompt=inference_prompt,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    if not getattr(result, "images", None):
        raise RuntimeError("FLUX.2 returned no image")
    raw = result.images[0].convert("RGB")
    if raw.size != source.size:
        raise RuntimeError("FLUX.2 output canvas changed")
    return raw


def _precompose_for_alpha(record: Mapping[str, Any], raw: Image.Image) -> Image.Image:
    with Image.open(record["source"]["image"]) as opened:
        source = opened.convert("RGB")
    masks = {}
    for filename in ("edit_core.png", "transition_band.png"):
        with Image.open(record["mask_bundle"]["assets"][filename]["path"]) as opened:
            masks[filename] = opened.convert("L")
    candidate, _ = semantic_masks.feathered_composite(
        source,
        raw,
        masks["edit_core.png"],
        masks["transition_band.png"],
    )
    return candidate


ISNET_SUBPROCESS_SOURCE = """
import sys
from PIL import Image
from rembg import new_session, remove
source = Image.open(sys.argv[1]).convert('RGB')
session = new_session('isnet-general-use')
mask = remove(source, session=session, only_mask=True, post_process_mask=True)
mask.convert('L').save(sys.argv[2], format='PNG')
""".strip()


def default_isnet_alpha_predictor(
    candidate: Image.Image, isnet_record: Mapping[str, Any]
) -> Image.Image:
    """Run the pinned local CPU ISNet environment without network access."""
    environment = isnet_record.get("environment")
    model = isnet_record.get("model")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(model, Mapping)
        or model.get("path") != str(ISNET_MODEL_PATH)
        or model.get("sha256") != ISNET_MODEL_SHA256
    ):
        raise ValueError("ISNet predictor requires the authenticated preflight record")
    python = Path(environment["python"])
    with tempfile.TemporaryDirectory(prefix="avengine_isnet_alpha_") as temporary:
        temporary_root = Path(temporary)
        input_path = temporary_root / "candidate.png"
        output_path = temporary_root / "alpha.png"
        candidate.convert("RGB").save(input_path, format="PNG")
        child_env = dict(os.environ)
        child_env.update(
            {
                "U2NET_HOME": str(ISNET_MODEL_PATH.parent),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        result = subprocess.run(
            [str(python), "-c", ISNET_SUBPROCESS_SOURCE, str(input_path), str(output_path)],
            env=child_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pinned ISNet alpha subprocess failed: {result.stderr}")
        if not output_path.is_file():
            raise RuntimeError("pinned ISNet alpha subprocess produced no output")
        with Image.open(output_path) as opened:
            opened.load()
            alpha = opened.convert("L")
        if alpha.size != candidate.size or alpha.getextrema() == (0, 0):
            raise RuntimeError("pinned ISNet alpha output is empty or has a wrong canvas")
        return alpha


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = preflight_batch(args.jobs_json, args.output_root, authenticate_model=True)
    pipeline = load_pipeline(preflight)
    results = run_preflighted_jobs(preflight, args.output_root, pipeline)
    print(json.dumps({"schema": "flux2_human_attribute_batch_result_v1", "results": results}, indent=2))
    successful = {"generated", "existing_success"}
    return 0 if all(item["status"] in successful for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
