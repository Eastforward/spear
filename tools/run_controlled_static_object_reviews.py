#!/usr/bin/env python3
"""Render authenticated multiview evidence for controlled static Pixal assets."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_pixal_jobs as pixal_runner


REVIEW_BATCH_SCHEMA = "avengine_controlled_static_object_review_batch_v1"
REVIEW_SCHEMA = "avengine_controlled_static_object_review_v1"
STATIC_ASSET_CLASS = "static_object"
STATIC_ROUTE = "flux2_pixal3d_static_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
RENDERER = SPEAR_ROOT / "tools/blender_render_i23d_review.py"
RENDER_VIEWS = ("front", "back", "side", "top", "quarter")
VIEW_RECORD_KEYS = {
    "orbit_anchor": "front",
    "orbit_opposite": "back",
    "orbit_right": "side",
    "orbit_top": "top",
    "orbit_quarter": "quarter",
}
RAW_PBR_MODE = "raw_glb_material"
CLAY_MODE = "neutral_clay_geometry_qa_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _relative(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_artifact(
    record: Any,
    *,
    base: Path,
    label: str,
    require_within_base: bool,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} record is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise contracts.ContractError(f"{label} path is invalid")
    unresolved = Path(raw_path)
    if not unresolved.is_absolute():
        unresolved = base / unresolved
    if unresolved.is_symlink():
        raise contracts.ContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    if require_within_base:
        try:
            path.relative_to(base.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"{label} escaped its immutable root"
            ) from error
    if (
        not path.is_file()
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] <= 0
        or path.stat().st_size != record["size_bytes"]
        or _sha256_file(path)
        != _require_sha256(record.get("sha256"), f"{label} hash")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _orientation_contract() -> dict[str, Any]:
    return {
        "reference_facing": {
            "role": "appearance_category_and_emitter_reference",
            "source_view": "generated_three_quarter_product_view",
            "canonical_heading_authority": False,
        },
        "review_orbit": {
            "frame": "source_glb_axes_review_only",
            "anchor_axis": "negative-y",
            "up_axis": "positive-z",
            "canonical_heading_authority": False,
            "view_mapping": copy.deepcopy(VIEW_RECORD_KEYS),
        },
        "canonical_heading": {
            "status": "deferred_to_static_finalization",
            "axis": None,
            "derived_from_reference_facing": False,
        },
    }


def _physical_scale_contract(target: Any) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise contracts.ContractError("static target physical profile is invalid")
    control = target.get("control_attribute")
    if control is not None and (not isinstance(control, str) or not control):
        raise contracts.ContractError(
            "physical control_attribute must be null or an ID"
        )
    measurement = target.get("measurement")
    if measurement != "height_cm":
        raise contracts.ContractError(
            "static finalization currently requires height_cm"
        )
    for field in ("target_value_cm", "tolerance_cm"):
        value = target.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise contracts.ContractError(
                f"static physical {field} must be finite and positive"
            )
    return {
        "status": "deferred_to_finalization",
        "control_attribute": control,
        "measurement": measurement,
        "target_physical_profile": copy.deepcopy(dict(target)),
    }


def _load_pixal_inputs_record(
    batch: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    record = batch.get("pixal_inputs")
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "manifest_sha256",
    }:
        raise contracts.ContractError("Pixal input binding is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise contracts.ContractError("Pixal input path is invalid")
    unresolved = Path(raw_path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError("Pixal input manifest is missing or unsafe")
    path = unresolved.resolve()
    if _sha256_file(path) != _require_sha256(
        record.get("sha256"), "Pixal input file hash"
    ):
        raise contracts.ContractError("Pixal input manifest file changed")
    loaded_path, payload = pixal_runner.load_pixal_inputs(path)
    if loaded_path != path:
        raise contracts.ContractError("Pixal input path resolution changed")
    if payload.get("manifest_sha256") != _require_sha256(
        record.get("manifest_sha256"), "Pixal input content hash"
    ):
        raise contracts.ContractError("Pixal input manifest content changed")
    return path, payload


def _validate_static_input_job(job: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(job, Mapping):
        raise contracts.ContractError("static Pixal input job must be an object")
    controlled = job.get("controlled_request")
    if not isinstance(controlled, Mapping):
        raise contracts.ContractError("static Pixal controlled request is missing")
    instance_id = controlled.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise contracts.ContractError("static Pixal instance ID is invalid")
    if (
        job.get("asset_class") != STATIC_ASSET_CLASS
        or job.get("route") != STATIC_ROUTE
        or controlled.get("asset_class") != STATIC_ASSET_CLASS
        or controlled.get("route") != STATIC_ROUTE
    ):
        raise contracts.ContractError(
            "Pixal input is not authenticated for the static-object route"
        )
    if "rig_mode" in job or controlled.get("rig_profile") is not None:
        raise contracts.ContractError("static Pixal input must not carry a rig binding")
    if job.get("legacy_tag") != instance_id:
        raise contracts.ContractError("static Pixal job identity changed")
    _require_sha256(controlled.get("request_sha256"), "static request hash")
    _require_sha256(controlled.get("profile_sha256"), "static profile hash")
    if (
        not isinstance(controlled.get("profile_schema_id"), str)
        or not controlled["profile_schema_id"]
        or not isinstance(controlled.get("sampled_attributes"), Mapping)
    ):
        raise contracts.ContractError("static Pixal request/profile binding is invalid")
    _physical_scale_contract(controlled.get("target_physical_profile"))
    return instance_id, controlled


def _live_mesh_readback(path: Path, expected: Any) -> dict[str, Any]:
    try:
        live = audit_mesh_efficiency.mesh_stats(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise contracts.ContractError("static Pixal GLB readback failed") from error
    if not isinstance(live, dict) or live.get("exists") is not True:
        raise contracts.ContractError("static Pixal GLB is unreadable")
    readback = {
        key: value
        for key, value in live.items()
        if key not in {"path", "exists"}
    }
    if not isinstance(expected, Mapping) or dict(expected) != readback:
        raise contracts.ContractError("static Pixal mesh readback changed")
    if (
        readback.get("vertices", 0) <= 0
        or readback.get("triangles", 0) <= 0
        or readback.get("materials", 0) <= 0
        or readback.get("textures", 0) <= 0
        or readback.get("skins") != 0
        or readback.get("animations") != 0
    ):
        raise contracts.ContractError(
            "static Pixal GLB must be textured, unskinned, and unanimated"
        )
    return readback


def load_static_pixal_batch(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    """Authenticate a Pixal batch and bind every attempt to one static input job."""

    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"static Pixal batch is missing: {unresolved}")
    path = unresolved.resolve()
    batch = contracts.load_json(path)
    if (
        not isinstance(batch, dict)
        or batch.get("schema") != pixal_runner.BATCH_SCHEMA
        or batch.get("status") != "passed_generation_and_glb_readback"
        or batch.get("state_classification") != "research_candidate"
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
        or batch.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(batch.get("attempts"), list)
        or batch.get("job_count") != len(batch["attempts"])
        or batch.get("passed_count") != len(batch["attempts"])
        or batch.get("failed_count") != 0
        or not batch["attempts"]
    ):
        raise contracts.ContractError("static Pixal batch contract/hash is invalid")

    _input_path, input_payload = _load_pixal_inputs_record(batch)
    if (
        input_payload.get("asset_class") != STATIC_ASSET_CLASS
        or input_payload.get("route") != STATIC_ROUTE
        or input_payload.get("formal_dataset_registration_authorized") is not False
        or input_payload.get("automatic_checks", {}).get(
            "static_jobs_have_no_rig_or_animation_binding"
        )
        is not True
    ):
        raise contracts.ContractError("Pixal input manifest is not static-object-only")

    input_jobs: dict[str, Mapping[str, Any]] = {}
    for job in input_payload["jobs"]:
        instance_id, _controlled = _validate_static_input_job(job)
        if instance_id in input_jobs:
            raise contracts.ContractError("duplicate static Pixal input instance")
        input_jobs[instance_id] = job

    root = path.parent
    bindings: dict[str, dict[str, Any]] = {}
    for attempt in batch["attempts"]:
        if not isinstance(attempt, Mapping):
            raise contracts.ContractError("static Pixal attempt must be an object")
        instance_id = attempt.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in bindings
            or instance_id not in input_jobs
        ):
            raise contracts.ContractError(
                "static Pixal attempt coverage/identity changed"
            )
        job = input_jobs[instance_id]
        controlled = job["controlled_request"]
        expected_bindings = {
            "execution_job_id": controlled["execution_job_id"],
            "request_sha256": controlled["request_sha256"],
            "profile_schema_id": controlled["profile_schema_id"],
            "sampled_attributes": controlled["sampled_attributes"],
            "target_physical_profile": controlled["target_physical_profile"],
            "seed": job["seed"],
            "attempt_ordinal": 0,
            "pixal_input": job["reference"]["pixal_input"],
        }
        for key, value in expected_bindings.items():
            if attempt.get(key) != value:
                raise contracts.ContractError(
                    f"static Pixal attempt {key} differs from its authenticated input"
                )

        glb = _resolve_artifact(
            attempt.get("output"),
            base=root,
            label=f"{instance_id} Pixal GLB",
            require_within_base=True,
        )
        if glb.suffix.lower() != ".glb" or Path(job.get("output", "")).resolve() != glb:
            raise contracts.ContractError("static Pixal output path/format changed")
        readback = _live_mesh_readback(glb, attempt.get("mesh_readback"))

        attempt_manifest_path = _resolve_artifact(
            attempt.get("attempt_manifest"),
            base=root,
            label=f"{instance_id} Pixal attempt manifest",
            require_within_base=True,
        )
        if Path(job.get("manifest", "")).resolve() != attempt_manifest_path:
            raise contracts.ContractError(
                "static Pixal attempt manifest path changed"
            )
        attempt_manifest = contracts.load_json(attempt_manifest_path)
        if (
            not isinstance(attempt_manifest, dict)
            or attempt_manifest.get("backend") != "pixal3d"
            or attempt_manifest.get("controlled_request") != controlled
            or attempt_manifest.get("output", {}).get("sha256")
            != attempt["output"]["sha256"]
            or Path(attempt_manifest.get("output", {}).get("path", "")).resolve()
            != glb
        ):
            raise contracts.ContractError(
                "static Pixal attempt manifest/request binding changed"
            )
        bindings[instance_id] = {
            "attempt": copy.deepcopy(dict(attempt)),
            "input_job": copy.deepcopy(dict(job)),
            "controlled_request": copy.deepcopy(dict(controlled)),
            "glb": glb,
            "mesh_readback": readback,
        }
    if set(bindings) != set(input_jobs):
        raise contracts.ContractError(
            "static Pixal batch does not cover every input job"
        )
    return path, batch, bindings


def _renderer_command(
    glb: Path, output_dir: Path, *, clay_geometry: bool
) -> list[str]:
    command = [
        str(BLENDER),
        "-b",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(glb),
        "--output-dir",
        str(output_dir),
        "--width",
        "480",
        "--height",
        "480",
        "--front-axis",
        "negative-y",
        "--include-top",
    ]
    if clay_geometry:
        command.append("--clay-preview")
    return command


def _render_pass(
    glb: Path,
    output_dir: Path,
    log_path: Path,
    *,
    clay_geometry: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    expected_mode = CLAY_MODE if clay_geometry else RAW_PBR_MODE
    command = _renderer_command(glb, output_dir, clay_geometry=clay_geometry)
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise contracts.ContractError(f"static-object Blender render failed: {glb}")

    manifest_path = output_dir / "render_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise contracts.ContractError("static-object render manifest is missing")
    manifest = contracts.load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or Path(manifest.get("input", "")).resolve() != glb
        or manifest.get("front_axis") != "negative-y"
        or set(manifest.get("views", {})) != set(RENDER_VIEWS)
        or manifest.get("resolution") != [480, 480]
        or manifest.get("material_preview", {}).get("mode") != expected_mode
    ):
        raise contracts.ContractError("static-object render manifest contract changed")
    images: dict[str, Path] = {}
    for record_key, renderer_key in VIEW_RECORD_KEYS.items():
        image = output_dir / f"{renderer_key}.png"
        if image.is_symlink() or not image.is_file():
            raise contracts.ContractError(
                f"static-object render is missing {renderer_key}"
            )
        with Image.open(image) as opened:
            opened.load()
            if opened.size != (480, 480):
                raise contracts.ContractError("static-object view resolution changed")
        images[record_key] = image
    return manifest, images


def _panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle(
        (8, 8, min(312, bounds[2] + 18), bounds[3] + 18), fill=(0, 0, 0)
    )
    draw.text((13, 13), label, font=font, fill=(255, 255, 255))
    return panel


def build_contact_sheet(
    reference_path: Path,
    raw_views: Mapping[str, Path],
    output: Path,
    clay_views: Mapping[str, Path] | None = None,
) -> None:
    with Image.open(reference_path) as opened:
        opened.load()
        reference = opened.convert("RGBA")
    backdrop = Image.new("RGB", reference.size, (205, 205, 205))
    backdrop.paste(reference.convert("RGB"), mask=reference.getchannel("A"))
    panels: list[tuple[str, Image.Image]] = [
        ("reference-facing; heading unknown", backdrop)
    ]
    labels = {
        "orbit_anchor": "raw PBR orbit anchor",
        "orbit_opposite": "raw PBR orbit opposite",
        "orbit_right": "raw PBR orbit right",
        "orbit_top": "raw PBR orbit top",
        "orbit_quarter": "raw PBR orbit quarter",
    }
    for key in VIEW_RECORD_KEYS:
        with Image.open(raw_views[key]) as opened:
            opened.load()
            panels.append((labels[key], opened.convert("RGB")))
    if clay_views is not None:
        for key in VIEW_RECORD_KEYS:
            with Image.open(clay_views[key]) as opened:
                opened.load()
                panels.append((f"clay {key.replace('_', ' ')}", opened.convert("RGB")))
    rows = (len(panels) + 2) // 3
    canvas = Image.new("RGB", (960, rows * 320), (28, 28, 28))
    for index, (label, image) in enumerate(panels):
        canvas.paste(_panel(image, label), ((index % 3) * 320, (index // 3) * 320))
    canvas.save(output, format="PNG", optimize=False, compress_level=6)


def _render_one(
    binding: Mapping[str, Any],
    staging: Path,
    *,
    include_clay_geometry: bool,
) -> dict[str, Any]:
    attempt = binding["attempt"]
    controlled = binding["controlled_request"]
    instance_id = attempt["instance_id"]
    destination = staging / instance_id
    destination.mkdir(parents=True, exist_ok=False)

    raw_dir = destination / "raw_pbr_views"
    raw_log = destination / "raw_pbr_blender.log"
    _raw_manifest, raw_paths = _render_pass(
        binding["glb"], raw_dir, raw_log, clay_geometry=False
    )
    clay_paths: dict[str, Path] | None = None
    clay_log: Path | None = None
    if include_clay_geometry:
        clay_dir = destination / "clay_geometry_views"
        clay_log = destination / "clay_geometry_blender.log"
        _clay_manifest, clay_paths = _render_pass(
            binding["glb"], clay_dir, clay_log, clay_geometry=True
        )

    reference_record = attempt["pixal_input"]
    reference_path = _resolve_artifact(
        reference_record,
        base=staging,
        label=f"{instance_id} reference RGBA",
        require_within_base=False,
    )
    contact_path = destination / "contact_sheet.png"
    build_contact_sheet(reference_path, raw_paths, contact_path, clay_paths)

    review: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "instance_id": instance_id,
        "execution_job_id": attempt["execution_job_id"],
        "request_sha256": attempt["request_sha256"],
        "profile_schema_id": attempt["profile_schema_id"],
        "profile_sha256": controlled["profile_sha256"],
        "asset_class": STATIC_ASSET_CLASS,
        "route": STATIC_ROUTE,
        "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
        "target_physical_profile": copy.deepcopy(
            attempt["target_physical_profile"]
        ),
        "physical_scale": _physical_scale_contract(
            attempt["target_physical_profile"]
        ),
        "orientation": _orientation_contract(),
        "pixal_output": copy.deepcopy(attempt["output"]),
        "mesh_readback": copy.deepcopy(binding["mesh_readback"]),
        "reference_rgba": copy.deepcopy(reference_record),
        "raw_pbr_render_manifest": _relative(
            raw_dir / "render_manifest.json", staging
        ),
        "raw_pbr_views": {
            key: _relative(path, staging) for key, path in raw_paths.items()
        },
        "raw_pbr_blender_log": _relative(raw_log, staging),
        "clay_geometry": (
            {
                "status": "included",
                "render_manifest": _relative(
                    destination / "clay_geometry_views" / "render_manifest.json",
                    staging,
                ),
                "views": {
                    key: _relative(path, staging)
                    for key, path in (clay_paths or {}).items()
                },
                "blender_log": _relative(clay_log, staging),
            }
            if clay_paths is not None and clay_log is not None
            else {"status": "not_requested"}
        ),
        "contact_sheet": _relative(contact_path, staging),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "automatic_checks": {
            "pixal_batch_input_and_attempt_reauthenticated": True,
            "asset_class_and_static_route_reauthenticated": True,
            "no_rig_binding": True,
            "no_skins_or_animations": True,
            "reference_and_raw_pbr_five_views_present": True,
            "reference_facing_not_used_as_canonical_heading": True,
            "physical_scale_deferred_to_finalization": True,
            "overall": "passed",
        },
        "visual_qa": "pending",
        "next_gate": "static_object_visual_decision",
    }
    review["review_sha256"] = _hash_without(review, "review_sha256")
    review_path = destination / "static_object_review_manifest.json"
    contracts.write_json_no_replace(review_path, review)
    return {
        "instance_id": instance_id,
        "request_sha256": attempt["request_sha256"],
        "review": _relative(review_path, staging),
        "review_sha256": review["review_sha256"],
        "contact_sheet": _relative(contact_path, staging),
        "status": "rendered_pending_visual_qa",
    }


def run_reviews(
    pixal_batch_path: Path,
    output_root: Path,
    workers: int,
    *,
    include_clay_geometry: bool = False,
) -> Path:
    pixal_batch_path, batch, bindings = load_static_pixal_batch(pixal_batch_path)
    if not 1 <= workers <= 4:
        raise contracts.ContractError("workers must be in [1, 4]")
    if not BLENDER.is_file() or not RENDERER.is_file():
        raise contracts.ContractError("pinned Blender/static renderer is missing")
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        reviews = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _render_one,
                    binding,
                    staging,
                    include_clay_geometry=include_clay_geometry,
                ): instance_id
                for instance_id, binding in bindings.items()
            }
            for future in as_completed(futures):
                review = future.result()
                reviews.append(review)
                print(
                    "CONTROLLED_STATIC_OBJECT_RENDERED "
                    f"instance={review['instance_id']}",
                    flush=True,
                )
        manifest: dict[str, Any] = {
            "schema": REVIEW_BATCH_SCHEMA,
            "status": "rendered_pending_visual_qa",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": STATIC_ASSET_CLASS,
            "route": STATIC_ROUTE,
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": batch["batch_sha256"],
            },
            "orientation": _orientation_contract(),
            "render_contract": {
                "reference_rgba_included": True,
                "raw_pbr_views": list(VIEW_RECORD_KEYS),
                "clay_geometry": (
                    "included" if include_clay_geometry else "not_requested"
                ),
            },
            "review_count": len(reviews),
            "reviews": sorted(reviews, key=lambda item: item["instance_id"]),
            "automatic_checks": {
                "all_pixal_outputs_and_requests_reauthenticated": True,
                "all_assets_static_unskinned_unanimated": True,
                "all_reference_and_raw_pbr_multiview_evidence_present": True,
                "all_visual_decisions_pending": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        }
        manifest["review_batch_sha256"] = _hash_without(
            manifest, "review_batch_sha256"
        )
        destination = staging / "static_object_review_batch_manifest.json"
        contracts.write_json_no_replace(destination, manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "static-object review output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / destination.name
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixal-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--include-clay-geometry",
        action="store_true",
        help="Add a neutral-clay five-view pass for ambiguous geometry.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = run_reviews(
            args.pixal_batch,
            args.output_root,
            args.workers,
            include_clay_geometry=args.include_clay_geometry,
        )
    except (
        contracts.ContractError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"CONTROLLED_STATIC_OBJECT_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(f"CONTROLLED_STATIC_OBJECT_REVIEW_OK output={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
