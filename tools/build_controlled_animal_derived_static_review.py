#!/usr/bin/env python3
"""Publish a pending-human static review for one repaired controlled animal.

The producer preserves the original raw-Pixal rejection, reauthenticates a
bounded same-source geometry repair, and renders a fresh material-mode
front/back/side/top/quarter review.  It cannot publish a decision,
``source_asset_v2``, source registry, UE job, or formal dataset admission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_pixal_jobs as pixal_runner

SPEAR_ROOT = Path(__file__).resolve().parents[1]
LOGICAL_TMP_ROOT = SPEAR_ROOT / "tmp"
PHYSICAL_TMP_ROOT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp"
)
RENDERER = SPEAR_ROOT / "tools/blender_render_i23d_review.py"
DEFAULT_BLENDER = Path("/data/jzy/.local/bin/blender").resolve()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLENDER_VERSION_RE = re.compile(r"^Blender\s+(.+?)\s*$")
BLENDER_BUILD_HASH_RE = re.compile(r"^\s*build hash:\s*(\S+)\s*$", re.IGNORECASE)
CLAIM_BOUNDARY = (
    "This pending-human review reauthenticates one frozen controlled request, "
    "its rejected raw Pixal output, a bounded same-source repair closure, the "
    "exact repaired GLB, inherited neutral-clay evidence, and a fresh PBR "
    "five-view render. It does not approve the repaired geometry or appearance, "
    "does not qualify PBR fidelity, and does not authorize source_asset_v2, a "
    "source registry, rigging, animation, UE execution, Native changes, or "
    "formal dataset admission."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _path_components(path: Path) -> list[Path]:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    result: list[Path] = []
    for part in absolute.parts[1:]:
        current = current / part
        result.append(current)
    return result


def _allowed_tmp_bridge(component: Path) -> bool:
    if component != LOGICAL_TMP_ROOT or not component.is_symlink():
        return False
    try:
        target = Path(os.readlink(component))
    except OSError:
        return False
    if not target.is_absolute():
        target = component.parent / target
    try:
        return os.path.samefile(target, PHYSICAL_TMP_ROOT)
    except OSError:
        return False


def _regular_file(path: Path, label: str) -> Path:
    absolute = Path(path).absolute()
    components = _path_components(absolute)
    for index, component in enumerate(components):
        if not component.is_symlink():
            continue
        is_leaf = index == len(components) - 1
        if _allowed_tmp_bridge(component) and not is_leaf:
            continue
        raise contracts.ContractError(
            f"{label} contains an unsafe symlink component: {component}"
        )
    if not absolute.is_file() or absolute.stat().st_size <= 0:
        raise contracts.ContractError(f"{label} is missing or empty: {absolute}")
    return absolute.resolve(strict=True)


def _external_file(path: Path, expected_sha256: str, label: str) -> Path:
    expected_sha256 = _require_sha256(expected_sha256, f"{label} expected SHA-256")
    result = _regular_file(path, label)
    if _sha256_file(result) != expected_sha256:
        raise contracts.ContractError(f"{label} changed from its external SHA-256")
    return result


def _new_output_root(path: Path) -> Path:
    absolute = Path(path).absolute()
    for component in _path_components(absolute):
        if component.is_symlink() and not _allowed_tmp_bridge(component):
            raise contracts.ContractError(
                f"output contains an unsafe symlink component: {component}"
            )
    if os.path.lexists(absolute):
        raise contracts.ContractError(f"refusing to replace output: {absolute}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    for component in _path_components(absolute.parent):
        if component.is_symlink() and not _allowed_tmp_bridge(component):
            raise contracts.ContractError(
                f"output contains an unsafe symlink component: {component}"
            )
    return absolute


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    payload = contracts.load_json(path)
    if not isinstance(payload, dict):
        raise contracts.ContractError(f"{label} must be a JSON object")
    return payload


def _absolute_record(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "recorded artifact")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = _regular_file(path, "review artifact")
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(
            "review artifact escaped its output root"
        ) from error
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _same_file(left: Path, right: Path, label: str) -> None:
    try:
        identical = os.path.samefile(left, right)
    except OSError as error:
        raise contracts.ContractError(
            f"cannot compare {label} file identity"
        ) from error
    if not identical:
        raise contracts.ContractError(f"{label} points to a different file")


def _descriptor_path(
    descriptor: Any,
    label: str,
    *,
    base: Path = SPEAR_ROOT,
    require_size: bool = True,
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise contracts.ContractError(f"{label} descriptor must be an object")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise contracts.ContractError(f"{label} descriptor path is invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    path = _regular_file(candidate, label)
    if _sha256_file(path) != _require_sha256(
        descriptor.get("sha256"), f"{label} SHA-256"
    ):
        raise contracts.ContractError(f"{label} SHA-256 changed")
    size = descriptor.get("size_bytes")
    if size is None and not require_size:
        return path
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise contracts.ContractError(f"{label} size changed")
    return path


def _descriptor_matches(
    descriptor: Any,
    expected: Path,
    label: str,
    *,
    base: Path = SPEAR_ROOT,
    require_size: bool = True,
) -> Path:
    path = _descriptor_path(
        descriptor,
        label,
        base=base,
        require_size=require_size,
    )
    _same_file(path, expected, label)
    return path


def _png(path: Path, label: str) -> Path:
    path = _regular_file(path, label)
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.format != "PNG" or opened.size != (480, 480):
                raise contracts.ContractError(f"{label} must be one 480x480 PNG")
    except OSError as error:
        raise contracts.ContractError(f"{label} is not a readable PNG") from error
    return path


def _load_pixal_batch(
    path: Path,
    *,
    requests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    payload = _strict_object(path, "Pixal batch")
    if (
        payload.get("schema") != pixal_runner.BATCH_SCHEMA
        or payload.get("status") != "passed_generation_and_glb_readback"
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("batch_sha256")
        != source_registry._hash_without(payload, "batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("Pixal batch contract/hash is invalid")
    inputs_descriptor = payload.get("pixal_inputs")
    inputs_path = _descriptor_path(
        inputs_descriptor,
        "Pixal inputs manifest",
        base=path.parent,
        require_size=False,
    )
    if (
        not isinstance(inputs_descriptor, Mapping)
        or inputs_descriptor.get("manifest_sha256") is None
    ):
        raise contracts.ContractError("Pixal inputs descriptor is incomplete")
    authenticated_path, inputs = pixal_runner.load_pixal_inputs(inputs_path)
    if authenticated_path != inputs_path or inputs.get(
        "manifest_sha256"
    ) != inputs_descriptor.get("manifest_sha256"):
        raise contracts.ContractError("Pixal inputs identity changed")
    jobs, attempts = source_registry.validate_pixal_request_identity(
        payload, inputs, requests
    )
    return payload, jobs, attempts, inputs_path


def _authenticate_source_authorities(
    *,
    instance_id: str,
    preflight_path: Path,
    expected_preflight_sha256: str,
    pixal_batch_path: Path,
    expected_pixal_batch_sha256: str,
    raw_static_decision_batch_path: Path,
    expected_raw_static_decision_batch_sha256: str,
) -> dict[str, Any]:
    preflight_path = _external_file(
        preflight_path, expected_preflight_sha256, "frozen preflight"
    )
    preflight, requests, profiles = source_registry.load_source_contract(
        preflight_path,
        frozen_historical_preflight=True,
    )
    request = requests.get(instance_id)
    if not isinstance(request, Mapping):
        raise contracts.ContractError(
            "selected instance is absent from the frozen request batch"
        )
    profile = profiles.get(request.get("profile_schema_id"))
    if not isinstance(profile, Mapping):
        raise contracts.ContractError("selected frozen profile is missing")

    pixal_batch_path = _external_file(
        pixal_batch_path, expected_pixal_batch_sha256, "Pixal batch"
    )
    pixal_batch, jobs, attempts, pixal_inputs_path = _load_pixal_batch(
        pixal_batch_path,
        requests=requests,
    )
    job = jobs.get(instance_id)
    attempt = attempts.get(instance_id)
    if not isinstance(job, Mapping) or not isinstance(attempt, Mapping):
        raise contracts.ContractError("selected Pixal attempt is missing")

    raw_static_decision_batch_path = _external_file(
        raw_static_decision_batch_path,
        expected_raw_static_decision_batch_sha256,
        "raw static decision batch",
    )
    (
        authenticated_decision_path,
        decision_batch,
        decisions,
    ) = source_registry.load_decision_batch(raw_static_decision_batch_path)
    if authenticated_decision_path != raw_static_decision_batch_path:
        raise contracts.ContractError("raw static decision batch identity changed")
    if set(decisions) != set(attempts):
        raise contracts.ContractError(
            "raw static decisions do not cover the Pixal attempt batch"
        )
    decision = decisions.get(instance_id)
    if not isinstance(decision, Mapping):
        raise contracts.ContractError("selected raw static decision is missing")
    decision_payload = decision.get("payload")
    if (
        not isinstance(decision_payload, Mapping)
        or decision_payload.get("decision") != "rejected"
        or decision_payload.get("state_classification") != "rejected"
        or decision_payload.get("formal_dataset_registration_authorized") is not False
        or decision_payload.get("next_gate") != "stop"
    ):
        raise contracts.ContractError(
            "derived review requires the original raw static rejection"
        )
    static_review = decision.get("static_review", {}).get("payload")
    if (
        not isinstance(static_review, Mapping)
        or static_review.get("instance_id") != instance_id
        or static_review.get("request_sha256") != request["request_sha256"]
        or static_review.get("pixal_output", {}).get("sha256")
        != attempt.get("output", {}).get("sha256")
    ):
        raise contracts.ContractError(
            "raw static rejection is not bound to the selected Pixal output"
        )

    raw_glb = _descriptor_path(
        attempt.get("output"),
        "raw Pixal GLB",
        base=pixal_batch_path.parent,
    )
    raw_attempt_manifest = _descriptor_path(
        attempt.get("attempt_manifest"),
        "raw Pixal attempt manifest",
        base=pixal_batch_path.parent,
    )
    reference = _descriptor_path(
        job.get("reference", {}).get("source"),
        "2D source reference",
    )
    pixal_input = _descriptor_path(
        attempt.get("pixal_input"),
        "Pixal RGBA input",
    )
    if contracts.canonical_json(attempt.get("pixal_input")) != contracts.canonical_json(
        job.get("reference", {}).get("pixal_input")
    ):
        raise contracts.ContractError("Pixal job/attempt RGBA identity changed")

    return {
        "preflight_path": preflight_path,
        "preflight": preflight,
        "request": copy.deepcopy(dict(request)),
        "profile": copy.deepcopy(dict(profile)),
        "pixal_batch_path": pixal_batch_path,
        "pixal_batch": pixal_batch,
        "pixal_inputs_path": pixal_inputs_path,
        "job": copy.deepcopy(dict(job)),
        "attempt": copy.deepcopy(dict(attempt)),
        "raw_glb": raw_glb,
        "raw_attempt_manifest": raw_attempt_manifest,
        "pixal_input": pixal_input,
        "reference": reference,
        "decision_batch_path": raw_static_decision_batch_path,
        "decision_batch": decision_batch,
        "decision_path": decision["path"],
        "decision": copy.deepcopy(dict(decision_payload)),
    }


def _validate_geometry_closure(
    *,
    closure_path: Path,
    repaired_glb: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    closure = _strict_object(closure_path, "derived geometry closure")
    if (
        closure.get("schema") != "avengine_generated_animal_geometry_closure_v1"
        or closure.get("status") != "pass_geometry_only"
        or closure.get("downstream", {}).get("formal_dataset_registration_authorized")
        is not False
        or closure.get("downstream", {}).get("ue_import_executed") is not False
    ):
        raise contracts.ContractError("derived geometry closure boundary changed")
    request = source["request"]
    if closure.get("candidate", {}).get("instance_id") != request["instance_id"]:
        raise contracts.ContractError("geometry closure instance identity changed")
    _descriptor_matches(
        closure.get("candidate", {}).get("source_pixal_glb"),
        source["raw_glb"],
        "geometry closure raw Pixal source",
    )
    _descriptor_matches(
        closure.get("candidate", {}).get("owner_approved_flux_reference"),
        source["reference"],
        "geometry closure 2D reference",
        require_size=False,
    )
    _descriptor_matches(
        closure.get("output", {}).get("glb"),
        repaired_glb,
        "geometry closure repaired GLB",
    )

    repair_manifest_path = _descriptor_path(
        closure.get("output", {}).get("repair_manifest"),
        "geometry repair manifest",
        require_size=False,
    )
    geometry_audit_path = _descriptor_path(
        closure.get("output", {}).get("independent_geometry_audit"),
        "independent geometry audit",
        require_size=False,
    )
    repair_manifest = _strict_object(repair_manifest_path, "geometry repair manifest")
    geometry_audit = _strict_object(geometry_audit_path, "independent geometry audit")
    untrusted_lineage = repair_manifest.get("lineage")
    if isinstance(untrusted_lineage, Mapping) and (
        untrusted_lineage.get("static_decision_state") != "rejected"
        or untrusted_lineage.get("raw_four_limbs_usable") is not False
        or untrusted_lineage.get("raw_pose_riggable") is not False
    ):
        raise contracts.ContractError("repair manifest upgraded the raw rejection")
    try:
        repair_manifest = review_contract.validate_bounded_repair_manifest(
            repair_manifest
        )
    except review_contract.DerivedStaticReviewContractError as error:
        raise contracts.ContractError(
            f"geometry repair manifest boundary changed: {error}"
        ) from error
    if (
        repair_manifest["lineage"]["instance_id"] != request["instance_id"]
    ):
        raise contracts.ContractError("geometry repair manifest identity changed")
    _descriptor_matches(
        repair_manifest.get("lineage", {}).get("pixal_source"),
        source["raw_glb"],
        "repair manifest raw Pixal source",
    )
    _descriptor_matches(
        repair_manifest.get("lineage", {}).get("pixal_manifest"),
        source["raw_attempt_manifest"],
        "repair manifest raw Pixal manifest",
    )
    _descriptor_matches(
        repair_manifest.get("lineage", {}).get("approved_reference"),
        source["reference"],
        "repair manifest 2D reference",
    )
    _descriptor_matches(
        repair_manifest.get("lineage", {}).get("static_decision"),
        source["decision_path"],
        "repair manifest raw static rejection",
    )
    _descriptor_path(
        repair_manifest.get("lineage", {}).get("owner_review"),
        "repair manifest owner review",
    )
    if (
        repair_manifest.get("lineage", {}).get("static_decision_state") != "rejected"
        or repair_manifest.get("lineage", {}).get("raw_four_limbs_usable") is not False
        or repair_manifest.get("lineage", {}).get("raw_pose_riggable") is not False
    ):
        raise contracts.ContractError("repair manifest upgraded the raw rejection")
    _descriptor_matches(
        repair_manifest.get("output"),
        repaired_glb,
        "geometry repair output",
    )

    repair = closure.get("repair")
    if (
        not isinstance(repair, Mapping)
        or repair.get("same_source_surface_only") is not True
        or repair.get("tail_region_selected_for_replacement_or_mirroring") is not False
        or any(
            repair.get(name) != []
            for name in (
                "external_geometry_inputs",
                "external_skeleton_inputs",
                "external_weight_inputs",
                "external_material_inputs",
                "external_texture_inputs",
                "animation_inputs",
            )
        )
    ):
        raise contracts.ContractError(
            "geometry closure is not a bounded same-source repair"
        )

    gates = closure.get("gates")
    automatic_statuses = {
        "four_independent_leg_chains": "passed",
        "no_low_cross_limb_membrane": "passed",
        "nonmanifold": "passed",
        "watertight": "passed",
    }
    inherited_statuses = {
        "single_breed_valid_tail": "passed_manual_multiview",
        "centerline": "passed_manual_top_view_review",
        "five_view_readback": "passed_manual_geometry_review",
    }
    if (
        not isinstance(gates, Mapping)
        or any(
            gates.get(name, {}).get("status") != expected
            for name, expected in automatic_statuses.items()
        )
        or any(
            gates.get(name, {}).get("status") != expected
            for name, expected in inherited_statuses.items()
        )
    ):
        raise contracts.ContractError("geometry closure gate statuses changed")
    five_view = gates["five_view_readback"]
    if (
        five_view.get("resolution") != [480, 480]
        or five_view.get("material_mode") != "neutral_clay_geometry_qa_v1"
        or not isinstance(five_view.get("checks"), Mapping)
        or not five_view["checks"]
        or any(value is not True for value in five_view["checks"].values())
        or set(five_view.get("views", {})) != set(review_contract.VIEWS)
    ):
        raise contracts.ContractError("clay five-view evidence is incomplete")
    clay_render_manifest_path = _descriptor_path(
        five_view.get("render_manifest"),
        "clay render manifest",
        require_size=False,
    )
    clay_render_manifest = _strict_object(
        clay_render_manifest_path, "clay render manifest"
    )
    if (
        Path(str(clay_render_manifest.get("input", ""))).resolve() != repaired_glb
        or clay_render_manifest.get("front_axis") not in review_contract.FRONT_AXES
        or clay_render_manifest.get("resolution") != [480, 480]
        or set(clay_render_manifest.get("views", {})) != set(review_contract.VIEWS)
        or clay_render_manifest.get("material_preview", {}).get("mode")
        != "neutral_clay_geometry_qa_v1"
    ):
        raise contracts.ContractError("clay render manifest changed")
    clay_views: dict[str, Path] = {}
    for view_name in review_contract.VIEWS:
        clay_views[view_name] = _png(
            _descriptor_path(
                five_view["views"][view_name],
                f"clay {view_name} view",
                require_size=False,
            ),
            f"clay {view_name} view",
        )
    clay_contact_sheet = _descriptor_path(
        five_view.get("contact_sheet"),
        "clay contact sheet",
    )
    try:
        with Image.open(clay_contact_sheet) as opened:
            opened.load()
            if opened.format != "PNG":
                raise contracts.ContractError("clay contact sheet must be a PNG")
    except OSError as error:
        raise contracts.ContractError("clay contact sheet is unreadable") from error

    readback = closure.get("container_readback")
    if (
        not isinstance(readback, Mapping)
        or any(
            isinstance(readback.get(name), bool)
            or not isinstance(readback.get(name), int)
            or readback[name] <= 0
            for name in ("material_count", "texture_count", "image_count")
        )
        or readback.get("pbr_fidelity_qualified") is not False
        or readback.get("skin_count") != 0
        or readback.get("animation_count") != 0
    ):
        raise contracts.ContractError(
            "derived GLB PBR container/readback boundary changed"
        )
    audit_records = geometry_audit.get("records")
    if (
        geometry_audit.get("schema") != "avengine_quadruped_i23d_geometry_audit_v3"
        or not isinstance(audit_records, list)
        or len(audit_records) != 1
    ):
        raise contracts.ContractError("independent geometry audit contract changed")
    audit_mesh = audit_records[0].get("mesh")
    _descriptor_matches(
        {
            "path": audit_mesh.get("absolute_path")
            if isinstance(audit_mesh, Mapping)
            else None,
            "sha256": audit_mesh.get("sha256")
            if isinstance(audit_mesh, Mapping)
            else None,
            "size_bytes": audit_mesh.get("size_bytes")
            if isinstance(audit_mesh, Mapping)
            else None,
        },
        repaired_glb,
        "independent geometry audit mesh",
    )

    return {
        "closure": closure,
        "repair_manifest_path": repair_manifest_path,
        "geometry_audit_path": geometry_audit_path,
        "repair_method": repair_manifest["implementation_contract"],
        "automatic_statuses": automatic_statuses,
        "inherited_statuses": {
            "single_breed_valid_tail": inherited_statuses["single_breed_valid_tail"],
            "centerline": inherited_statuses["centerline"],
            "clay_five_view": inherited_statuses["five_view_readback"],
        },
        "clay": {
            "front_axis": clay_render_manifest["front_axis"],
            "render_manifest_path": clay_render_manifest_path,
            "views": clay_views,
            "contact_sheet": clay_contact_sheet,
        },
        "pbr_container": {
            name: readback[name]
            for name in (
                "material_count",
                "texture_count",
                "image_count",
                "pbr_fidelity_qualified",
            )
        },
    }


def _panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bounds[2] + 18, bounds[3] + 18), fill=(0, 0, 0))
    draw.text((13, 13), label, font=font, fill=(255, 255, 255))
    return panel


def _build_contact_sheet(
    reference_path: Path,
    view_paths: Mapping[str, Path],
    output_path: Path,
) -> None:
    with Image.open(reference_path) as opened:
        opened.load()
        reference = opened.convert("RGBA")
    backdrop = Image.new("RGB", reference.size, (205, 205, 205))
    backdrop.paste(reference.convert("RGB"), mask=reference.getchannel("A"))
    panels: list[tuple[str, Image.Image]] = [("frozen 2D reference", backdrop)]
    for name in review_contract.VIEWS:
        with Image.open(view_paths[name]) as opened:
            opened.load()
            panels.append((f"PBR {name}", opened.convert("RGB")))
    canvas = Image.new("RGB", (960, 640), (28, 28, 28))
    for index, (label, image) in enumerate(panels):
        canvas.paste(_panel(image, label), ((index % 3) * 320, (index // 3) * 320))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False, compress_level=6)


def _blender_identity(blender: Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(blender), "--version"],
        cwd=SPEAR_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise contracts.ContractError("cannot read pinned Blender identity")
    version = None
    build_hash = None
    for line in completed.stdout.splitlines():
        version_match = BLENDER_VERSION_RE.fullmatch(line)
        build_match = BLENDER_BUILD_HASH_RE.fullmatch(line)
        if version_match:
            version = version_match.group(1)
        if build_match:
            build_hash = build_match.group(1)
    if not version or not build_hash:
        raise contracts.ContractError("Blender version/build hash is incomplete")
    return {
        "path": str(blender),
        "version": version,
        "build_hash": build_hash,
    }


def _run_pbr_review(
    *,
    repaired_glb: Path,
    reference: Path,
    front_axis: str,
    staging: Path,
    blender: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    blender = _regular_file(blender, "Blender executable")
    renderer = _regular_file(RENDERER, "PBR review renderer")
    pbr_root = staging / "pbr_five_view"
    view_root = pbr_root / "views"
    log_path = pbr_root / "blender.log"
    pbr_root.mkdir(parents=True, exist_ok=False)
    command = [
        str(blender),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(renderer),
        "--",
        "--input",
        str(repaired_glb),
        "--output-dir",
        str(view_root),
        "--width",
        "480",
        "--height",
        "480",
        "--samples",
        "16",
        "--include-top",
        "--animal-material-preview",
        "--front-axis",
        front_axis,
    ]
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
        raise contracts.ContractError("derived-static PBR five-view render failed")
    render_manifest_path = _regular_file(
        view_root / "render_manifest.json", "PBR render manifest"
    )
    render_manifest = _strict_object(render_manifest_path, "PBR render manifest")
    if (
        Path(str(render_manifest.get("input", ""))).resolve() != repaired_glb
        or render_manifest.get("front_axis") != front_axis
        or render_manifest.get("resolution") != [480, 480]
        or render_manifest.get("samples") != 16
        or set(render_manifest.get("views", {})) != set(review_contract.VIEWS)
        or render_manifest.get("material_preview", {}).get("mode")
        != "ue_animal_nonmetallic_roughness_preview_v1"
        or render_manifest.get("material_preview", {}).get(
            "principled_nodes_changed", 0
        )
        <= 0
        or render_manifest.get("lighting")
        != {
            "area_light_scale": 0.25,
            "world_strength": 0.25,
            "exposure": -0.5,
        }
    ):
        raise contracts.ContractError("derived-static PBR render manifest changed")
    views = {
        name: _png(view_root / f"{name}.png", f"PBR {name} view")
        for name in review_contract.VIEWS
    }
    contact_sheet = pbr_root / "contact_sheet.png"
    _build_contact_sheet(reference, views, contact_sheet)
    _regular_file(contact_sheet, "PBR contact sheet")
    return (
        {
            "front_axis": front_axis,
            "resolution": [480, 480],
            "material_mode": "ue_animal_nonmetallic_roughness_preview_v1",
            "render_manifest": _relative_record(render_manifest_path, staging),
            "views": {
                name: _relative_record(path, staging)
                for name, path in sorted(views.items())
            },
            "contact_sheet": _relative_record(contact_sheet, staging),
            "execution_log": _relative_record(log_path, staging),
        },
        _blender_identity(blender),
    )


def publish_review(
    *,
    instance_id: str,
    preflight_path: Path,
    expected_preflight_sha256: str,
    pixal_batch_path: Path,
    expected_pixal_batch_sha256: str,
    raw_static_decision_batch_path: Path,
    expected_raw_static_decision_batch_sha256: str,
    geometry_closure_path: Path,
    expected_geometry_closure_sha256: str,
    repaired_glb_path: Path,
    expected_repaired_glb_sha256: str,
    output_root: Path,
    blender: Path = DEFAULT_BLENDER,
) -> Path:
    source = _authenticate_source_authorities(
        instance_id=instance_id,
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        pixal_batch_path=pixal_batch_path,
        expected_pixal_batch_sha256=expected_pixal_batch_sha256,
        raw_static_decision_batch_path=raw_static_decision_batch_path,
        expected_raw_static_decision_batch_sha256=(
            expected_raw_static_decision_batch_sha256
        ),
    )
    geometry_closure_path = _external_file(
        geometry_closure_path,
        expected_geometry_closure_sha256,
        "derived geometry closure",
    )
    repaired_glb = _external_file(
        repaired_glb_path,
        expected_repaired_glb_sha256,
        "repaired GLB",
    )
    geometry = _validate_geometry_closure(
        closure_path=geometry_closure_path,
        repaired_glb=repaired_glb,
        source=source,
    )
    output_root = _new_output_root(output_root)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".staging",
            dir=output_root.parent,
        )
    )
    try:
        pbr_evidence, blender_identity = _run_pbr_review(
            repaired_glb=repaired_glb,
            reference=source["reference"],
            front_axis=geometry["clay"]["front_axis"],
            staging=staging,
            blender=blender,
        )
        request = source["request"]
        decision = source["decision"]
        clay = geometry["clay"]
        manifest: dict[str, Any] = {
            "schema": review_contract.REVIEW_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": review_contract.REVIEW_STATUS,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "instance_identity": {
                "instance_id": request["instance_id"],
                "profile_schema_id": request["profile_schema_id"],
                "profile_sha256": request["profile_sha256"],
                "request_sha256": request["request_sha256"],
                "taxonomy": copy.deepcopy(request["taxonomy"]),
                "fixed_attributes": copy.deepcopy(request["fixed_attributes"]),
                "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
                "target_physical_profile": copy.deepcopy(
                    request["target_physical_profile"]
                ),
            },
            "source_authorities": {
                "frozen_preflight": {
                    "file": _absolute_record(source["preflight_path"]),
                    "preflight_sha256": source["preflight"]["preflight_sha256"],
                    "validation_mode": "frozen_historical_preflight_v1",
                },
                "pixal_batch": {
                    "file": _absolute_record(source["pixal_batch_path"]),
                    "batch_sha256": source["pixal_batch"]["batch_sha256"],
                },
                "raw_static_decision_batch": {
                    "file": _absolute_record(source["decision_batch_path"]),
                    "decision_batch_sha256": source["decision_batch"][
                        "decision_batch_sha256"
                    ],
                },
                "raw_static_decision": {
                    "file": _absolute_record(source["decision_path"]),
                    "decision_sha256": decision["decision_sha256"],
                    "decision": "rejected",
                    "state_classification": "rejected",
                    "formal_dataset_registration_authorized": False,
                },
                "raw_pixal_glb": _absolute_record(source["raw_glb"]),
                "reference_2d": _absolute_record(source["reference"]),
            },
            "derived_geometry": {
                "geometry_closure": _absolute_record(geometry_closure_path),
                "repaired_glb": _absolute_record(repaired_glb),
                "repair_manifest": _absolute_record(geometry["repair_manifest_path"]),
                "independent_geometry_audit": _absolute_record(
                    geometry["geometry_audit_path"]
                ),
                "repair_method": geometry["repair_method"],
                "lineage_kind": "bounded_same_pixal_mesh_repair",
                "automatic_gate_statuses": geometry["automatic_statuses"],
                "inherited_manual_review_statuses": geometry["inherited_statuses"],
                "pbr_container_readback": geometry["pbr_container"],
            },
            "evidence": {
                "clay_five_view": {
                    "front_axis": clay["front_axis"],
                    "resolution": [480, 480],
                    "material_mode": "neutral_clay_geometry_qa_v1",
                    "render_manifest": _absolute_record(clay["render_manifest_path"]),
                    "views": {
                        name: _absolute_record(path)
                        for name, path in sorted(clay["views"].items())
                    },
                    "contact_sheet": _absolute_record(clay["contact_sheet"]),
                },
                "pbr_five_view": pbr_evidence,
            },
            "producer": {
                "tool": _absolute_record(Path(__file__)),
                "renderer": _absolute_record(RENDERER),
                "blender": blender_identity,
            },
            "automatic_checks": {
                name: True for name in sorted(review_contract.AUTOMATIC_CHECK_FIELDS)
            },
            "human_review": {
                "status": "pending",
                "decision": None,
                "checks": {
                    name: None for name in sorted(review_contract.HUMAN_CHECK_FIELDS)
                },
                "review_authority_required": (
                    "explicit_project_owner_decision_bound_to_review_file_sha256"
                ),
                "decision_artifact": None,
            },
            "next_gate": review_contract.NEXT_GATE,
        }
        manifest["review_sha256"] = review_contract.hash_without(
            manifest, "review_sha256"
        )
        review_contract.validate_review(manifest)
        manifest_path = contracts.write_json_no_replace(
            staging / "derived_static_review.json", manifest
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "derived-static review output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / manifest_path.name
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--pixal-batch", required=True, type=Path)
    parser.add_argument("--expected-pixal-batch-sha256", required=True)
    parser.add_argument("--raw-static-decision-batch", required=True, type=Path)
    parser.add_argument("--expected-raw-static-decision-batch-sha256", required=True)
    parser.add_argument("--geometry-closure", required=True, type=Path)
    parser.add_argument("--expected-geometry-closure-sha256", required=True)
    parser.add_argument("--repaired-glb", required=True, type=Path)
    parser.add_argument("--expected-repaired-glb-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = publish_review(
            instance_id=args.instance_id,
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            pixal_batch_path=args.pixal_batch,
            expected_pixal_batch_sha256=args.expected_pixal_batch_sha256,
            raw_static_decision_batch_path=args.raw_static_decision_batch,
            expected_raw_static_decision_batch_sha256=(
                args.expected_raw_static_decision_batch_sha256
            ),
            geometry_closure_path=args.geometry_closure,
            expected_geometry_closure_sha256=(args.expected_geometry_closure_sha256),
            repaired_glb_path=args.repaired_glb,
            expected_repaired_glb_sha256=args.expected_repaired_glb_sha256,
            output_root=args.output_root,
            blender=args.blender,
        )
        payload = review_contract.validate_review(contracts.load_json(manifest_path))
    except (
        contracts.ContractError,
        review_contract.DerivedStaticReviewContractError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(
            f"CONTROLLED_ANIMAL_DERIVED_STATIC_REVIEW_FAILED {error}", file=sys.stderr
        )
        return 2
    print(
        "CONTROLLED_ANIMAL_DERIVED_STATIC_REVIEW_OK "
        f"instance={payload['instance_identity']['instance_id']} "
        f"status={payload['status']} "
        f"review_sha256={payload['review_sha256']} "
        f"output={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
