#!/usr/bin/env python3
"""Generate FLUX.2 color-only feasibility canaries for fixed Route-2 templates.

These eight canaries deliberately test the expensive full 2D-to-3D path.  They
do not replace the production policy, where qualified geometry uses a
deterministic semantic material transform for ordinary color sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import human_attribute_masks as semantic_masks
from tools import route2_controlled_geometry_references_v3 as geometry


SCHEMA = "route2_controlled_color_reference_jobs_v3"
CANDIDATE_SCHEMA = "route2_controlled_color_reference_candidate_v3"
DECISION_SCHEMA = "route2_controlled_color_reference_agent_qa_v1"
PIXAL_JOBS_SCHEMA = "route2_controlled_color_pixal_jobs_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
OUTPUT_ROOT = SPEAR_ROOT / "tmp/route2_controlled_color_references_v3"
PIXAL_OUTPUT_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_color_v3/pixal3d"
ATTRIBUTES = ("top_color", "trousers_color", "fixed_shoes_color", "fixed_hair_color")


class ColorReferenceError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _case(
    case_id: str,
    sex: str,
    attribute: str,
    seed: int,
    color_name: str,
    color_rgb: tuple[int, int, int],
    mask: Mapping[str, Any],
) -> dict[str, Any]:
    subject = "man" if sex == "male" else "woman"
    target = {
        "top_color": "the existing short-sleeve T-shirt",
        "trousers_color": "the existing full-length trousers",
        "fixed_shoes_color": "the existing fixed low-top shoes",
        "fixed_hair_color": "the existing fixed hairstyle",
    }[attribute]
    prompt = (
        f"Edit Image 1, the approved full-body soft T-pose reference of the same adult {subject}. "
        f"Change exactly one color attribute: recolor {target} to natural {color_name} "
        f"sRGB rgb{color_rgb}. Preserve its exact geometry, silhouette, seams, wrinkles, texture, "
        "shading and material response. Preserve the face, skin, body, hairstyle geometry, pose, "
        "camera, limb gaps, every other garment, the shoes unless shoes are the target, and the "
        "identical light-gray background."
    )
    negative = (
        "geometry change, garment style change, sleeve length change, shorts, different shoes, "
        "different hairstyle, hat, glasses, identity change, face change, body change, pose change, "
        "camera change, crop, fused limbs, moved hands, logo, text, pattern, multiple people"
    )
    return {
        "case_id": case_id,
        "sex": sex,
        "base_asset_id": geometry.SOURCE_PINS[sex]["asset_id"],
        "attribute": attribute,
        "seed": seed,
        "target_color_name": color_name,
        "target_color_srgb": list(color_rgb),
        "prompt": prompt,
        "negative_prompt": negative,
        "mask": dict(mask),
        "alpha_policy": "source_alpha_byte_identical",
        "execution_policy": "flux2_color_feasibility_fallback_not_ordinary_production",
    }


TOP_MALE = {
    "strategy": "source_foreground_lab_connected_component",
    "roi": [0.25, 0.20, 0.75, 0.53],
    "seed_points": [[0.50, 0.32], [0.38, 0.31], [0.62, 0.31]],
    "lab_tolerance": 34.0,
    "radius": 4,
}
TOP_FEMALE = {
    "strategy": "source_foreground_lab_connected_component",
    "roi": [0.27, 0.20, 0.73, 0.51],
    "seed_points": [[0.50, 0.33], [0.39, 0.31], [0.61, 0.31]],
    "lab_tolerance": 38.0,
    "radius": 4,
}
TROUSERS_MALE = {
    "strategy": "source_foreground_lab_connected_component",
    "roi": [0.35, 0.50, 0.65, 0.89],
    "seed_points": [[0.44, 0.60], [0.56, 0.60], [0.44, 0.80], [0.56, 0.80]],
    "lab_tolerance": 38.0,
    "radius": 4,
}
TROUSERS_FEMALE = {
    "strategy": "source_foreground_lab_connected_component",
    "roi": [0.35, 0.50, 0.65, 0.89],
    "seed_points": [[0.44, 0.60], [0.56, 0.60], [0.44, 0.80], [0.56, 0.80]],
    "lab_tolerance": 40.0,
    "radius": 4,
}
SHOES = {
    "strategy": "source_foreground_lab_connected_component",
    "roi": [0.34, 0.875, 0.66, 0.97],
    "seed_points": [[0.415, 0.925], [0.585, 0.925]],
    "lab_tolerance": 40.0,
    "radius": 3,
}
HAIR_MALE = {
    "strategy": "reviewed_source_hair_polygons",
    "polygons": [
        [[0.446, 0.158], [0.447, 0.108], [0.462, 0.074], [0.485, 0.057],
         [0.520, 0.055], [0.546, 0.073], [0.560, 0.110], [0.554, 0.158],
         [0.530, 0.139], [0.500, 0.132], [0.470, 0.145]],
    ],
    "radius": 3,
}
HAIR_FEMALE = {
    "strategy": "reviewed_source_hair_polygons",
    "polygons": [
        [[0.444, 0.145], [0.449, 0.091], [0.466, 0.064], [0.492, 0.052],
         [0.522, 0.057], [0.548, 0.083], [0.555, 0.143], [0.530, 0.122],
         [0.500, 0.113], [0.468, 0.125]],
        [[0.532, 0.132], [0.558, 0.142], [0.573, 0.174], [0.572, 0.226],
         [0.548, 0.242], [0.537, 0.204]],
    ],
    "radius": 3,
}


CASE_SPECS = (
    _case("male_top_cobalt", "male", "top_color", 401, "muted cobalt blue", (36, 82, 154), TOP_MALE),
    _case("female_top_teal", "female", "top_color", 402, "muted teal", (40, 122, 120), TOP_FEMALE),
    _case("male_trousers_navy", "male", "trousers_color", 403, "deep navy", (39, 58, 89), TROUSERS_MALE),
    _case("female_trousers_khaki", "female", "trousers_color", 404, "warm khaki", (165, 138, 94), TROUSERS_FEMALE),
    _case("male_shoes_black", "male", "fixed_shoes_color", 405, "soft black", (32, 33, 36), SHOES),
    _case("female_shoes_brown", "female", "fixed_shoes_color", 406, "warm brown", (107, 77, 61), SHOES),
    _case("male_hair_auburn", "male", "fixed_hair_color", 407, "dark auburn", (91, 47, 37), HAIR_MALE),
    _case("female_hair_chestnut", "female", "fixed_hair_color", 408, "chestnut brown", (112, 69, 47), HAIR_FEMALE),
)
CASE_BY_ID = {case["case_id"]: case for case in CASE_SPECS}


def _record(path: Path, *, public_path: Path | None = None, mode: int | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        raise ColorReferenceError(f"artifact must be a direct nonempty file: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise ColorReferenceError(f"artifact mode changed: {path}")
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": geometry.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_edit_core(case: Mapping[str, Any], source: Image.Image, alpha: Image.Image) -> Image.Image:
    mask = case["mask"]
    foreground = np.asarray(alpha.convert("L"), dtype=np.uint8) >= 128
    if mask["strategy"] == "source_foreground_lab_connected_component":
        values = semantic_masks._seeded_lab_semantic(source, alpha, mask)
    elif mask["strategy"] == "reviewed_source_hair_polygons":
        values = np.asarray(geometry._polygon_mask(mask["polygons"]), dtype=bool) & foreground
        values = ndimage.binary_closing(values, iterations=1) & foreground
    else:
        raise ColorReferenceError(f"unknown mask strategy: {mask['strategy']}")
    if not np.any(values) or np.all(values):
        raise ColorReferenceError(f"empty/full color semantic mask: {case['case_id']}")
    return Image.fromarray(np.where(values, 255, 0).astype(np.uint8), "L")


def evaluate_metrics(
    case: Mapping[str, Any],
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
    core: Image.Image,
    band: Image.Image,
    pixel_proof: Mapping[str, Any],
) -> dict[str, Any]:
    left = np.asarray(source.convert("RGB"), dtype=np.uint8)
    right = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    core_values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    band_values = np.asarray(band.convert("L"), dtype=np.uint8) == 255
    guard = ~(core_values | band_values)
    left_lab = semantic_masks._srgb_to_lab(left)
    right_lab = semantic_masks._srgb_to_lab(right)
    delta_e = np.linalg.norm(right_lab - left_lab, axis=2)
    changed = delta_e >= 5.0
    changed_fraction = float(np.count_nonzero(changed & core_values) / np.count_nonzero(core_values))
    target = np.asarray(case["target_color_srgb"], dtype=np.uint8).reshape(1, 1, 3)
    target_lab = semantic_masks._srgb_to_lab(target)[0, 0]
    target_delta = float(np.linalg.norm(np.median(right_lab[core_values], axis=0) - target_lab))
    left_luma = left[..., 0] * 0.2126 + left[..., 1] * 0.7152 + left[..., 2] * 0.0722
    right_luma = right[..., 0] * 0.2126 + right[..., 1] * 0.7152 + right[..., 2] * 0.0722
    luma_correlation = semantic_masks._safe_correlation(left_luma[core_values], right_luma[core_values])
    left_edges = np.hypot(ndimage.sobel(left_luma, 0), ndimage.sobel(left_luma, 1))
    right_edges = np.hypot(ndimage.sobel(right_luma, 0), ndimage.sobel(right_luma, 1))
    edge_correlation = semantic_masks._safe_correlation(left_edges[core_values], right_edges[core_values])
    source_a = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    candidate_a = np.asarray(candidate_alpha.convert("L"), dtype=np.uint8)
    x = np.arange(geometry.WIDTH)[None, :]
    pair_fractions = []
    for side in (x < geometry.WIDTH / 2, x >= geometry.WIDTH / 2):
        region = core_values & side
        pair_fractions.append(float(np.count_nonzero(changed & region) / max(1, np.count_nonzero(region))))
    rows_source = np.where(source_a >= 128)[0]
    rows_candidate = np.where(candidate_a >= 128)[0]
    foot_delta = abs(int(rows_source.max()) - int(rows_candidate.max()))
    checks = {
        "outside_mask_rgb_exact": pixel_proof["outside_changed_pixels"] == 0
        and pixel_proof["outside_max_abs_channel_delta"] == 0,
        "source_alpha_byte_identical": bool(np.array_equal(source_a, candidate_a)),
        "non_target_guard_byte_identical": bool(np.array_equal(left[guard], right[guard])),
        "semantic_core_changed": changed_fraction >= 0.30,
        "natural_target_color_close": target_delta <= 45.0,
        "texture_luminance_retained": luma_correlation >= 0.50,
        "texture_edges_retained": edge_correlation >= 0.50,
        "floor_contact_unchanged": foot_delta == 0,
    }
    if case["attribute"] in {"trousers_color", "fixed_shoes_color"}:
        checks["bilateral_target_changed"] = min(pair_fractions) >= 0.25
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "pixel_proof": dict(pixel_proof),
            "semantic_core_changed_fraction": changed_fraction,
            "target_median_delta_e": target_delta,
            "luminance_correlation": luma_correlation,
            "edge_correlation": edge_correlation,
            "left_target_changed_fraction": pair_fractions[0],
            "right_target_changed_fraction": pair_fractions[1],
            "foot_contact_y_delta_px": foot_delta,
            "guard_pixels": int(np.count_nonzero(guard)),
        },
    }


def prepare() -> Path:
    output = OUTPUT_ROOT.absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    sources = {sex: geometry.authenticate_source(sex) for sex in ("male", "female")}
    model = geometry.authenticate_model()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        (staging / "masks").mkdir()
        (staging / "cases").mkdir()
        (staging / "failures").mkdir()
        cases = []
        for spec in CASE_SPECS:
            source = sources[spec["sex"]]
            with Image.open(source["image"]["path"]) as opened:
                image = opened.convert("RGB")
            with Image.open(source["alpha"]["path"]) as opened:
                alpha = opened.convert("L")
            core = build_edit_core(spec, image, alpha)
            band, guard = geometry.transition_and_guard(core, int(spec["mask"]["radius"]))
            mask_root = staging / "masks" / spec["case_id"]
            mask_root.mkdir()
            public_root = output / "masks" / spec["case_id"]
            images = {
                "edit_core.png": core,
                "transition_band.png": band,
                "protected_guard.png": guard,
                "overlay.png": geometry._mask_overlay(image, core, band),
            }
            for filename, value in images.items():
                geometry._write_image(mask_root / filename, value)
            assets = {
                name: _record(mask_root / name, public_path=public_root / name)
                for name in sorted(images)
            }
            manifest = {
                "schema": "route2_controlled_color_mask_bundle_v3",
                "case_id": spec["case_id"],
                "attribute": spec["attribute"],
                "source_image": source["image"],
                "source_alpha": source["alpha"],
                "construction": spec["mask"],
                "assets": assets,
                "metrics": {
                    "core_pixels": int(np.count_nonzero(np.asarray(core) == 255)),
                    "transition_pixels": int(np.count_nonzero(np.asarray(band) == 255)),
                    "protected_pixels": int(np.count_nonzero(np.asarray(guard) == 255)),
                    "exact_partition": True,
                },
            }
            mask_manifest = mask_root / "mask_manifest.json"
            mask_manifest.write_bytes(geometry._json_bytes(manifest))
            cases.append({
                **spec,
                "source": source,
                "mask_manifest": _record(mask_manifest, public_path=public_root / "mask_manifest.json"),
                "mask_assets": assets,
                "inference": {
                    "width": geometry.WIDTH,
                    "height": geometry.HEIGHT,
                    "steps": geometry.STEPS,
                    "guidance_scale": geometry.GUIDANCE_SCALE,
                    "max_sequence_length": geometry.MAX_SEQUENCE_LENGTH,
                    "local_files_only": True,
                },
            })
        payload = {
            "schema": SCHEMA,
            "state_classification": "research_candidate_preflight",
            "formal_registration_authorized": False,
            "purpose": "FLUX.2 color-only feasibility fallback through Pixal3D and TokenRig",
            "ordinary_production_color_policy": "deterministic_semantic_material_transform",
            "flux2_is_ordinary_production_color_backend": False,
            "prohibited_models": ["Hunyuan3D", "Qwen-Image", "FLUX.1", "other_image_models"],
            "output_root": str(output),
            "created_at_utc": _utc_now(),
            "runner": _record(RUNNER_PATH),
            "model": model,
            "sources": sources,
            "cases": cases,
        }
        (staging / "color_jobs_v3.json").write_bytes(geometry._json_bytes(payload))
        geometry._readonly_tree(staging)
        geometry._fsync_tree(staging)
        geometry._rename_noreplace(staging, output)
        return output / "color_jobs_v3.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_contract() -> dict[str, Any]:
    path = OUTPUT_ROOT / "color_jobs_v3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("formal_registration_authorized") is not False
        or payload.get("flux2_is_ordinary_production_color_backend") is not False
        or payload.get("runner") != _record(RUNNER_PATH)
        or [case.get("case_id") for case in payload.get("cases", [])] != list(CASE_BY_ID)
    ):
        raise ColorReferenceError("color contract schema/state/runner/case order changed")
    return payload


def _publish_failure(case_id: str, staging: Path, error: BaseException) -> Path:
    destination = OUTPUT_ROOT / "failures" / f"{case_id}.{uuid.uuid4().hex}"
    (staging / "failure.json").write_bytes(geometry._json_bytes({
        "schema": "route2_controlled_color_generation_failure_v1",
        "case_id": case_id,
        "state_classification": "rejected",
        "error": {"type": type(error).__name__, "message": str(error)},
        "recorded_at_utc": _utc_now(),
    }))
    geometry._readonly_tree(staging)
    geometry._fsync_tree(staging)
    geometry._rename_noreplace(staging, destination)
    return destination


def generate_case(contract: Mapping[str, Any], case_id: str, pipeline: Any, gpu: str) -> dict[str, Any]:
    matches = [case for case in contract["cases"] if case["case_id"] == case_id]
    if len(matches) != 1 or case_id not in CASE_BY_ID:
        raise ColorReferenceError(f"unknown/nonunique case: {case_id}")
    case = matches[0]
    destination = OUTPUT_ROOT / "cases" / case_id
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{case_id}.", suffix=".staging", dir=OUTPUT_ROOT / "cases"))
    try:
        with Image.open(case["source"]["image"]["path"]) as opened:
            source = opened.convert("RGB")
        with Image.open(case["source"]["alpha"]["path"]) as opened:
            source_alpha = opened.convert("L")
        mask_root = Path(case["mask_manifest"]["path"]).parent
        with Image.open(mask_root / "edit_core.png") as opened:
            core = opened.convert("L")
        with Image.open(mask_root / "transition_band.png") as opened:
            band = opened.convert("L")
        raw = geometry._inference(case, source, pipeline)
        candidate, pixel_proof = geometry.composite_candidate(source, raw, core, band)
        candidate_alpha = source_alpha.copy()
        metrics = evaluate_metrics(
            case, source, candidate, source_alpha, candidate_alpha, core, band, pixel_proof
        )
        rgba = candidate.convert("RGBA")
        rgba.putalpha(candidate_alpha)
        images = {
            "source.png": source,
            "source_alpha.png": source_alpha,
            "raw_candidate.png": raw,
            "candidate.png": candidate,
            "candidate_alpha.png": candidate_alpha,
            "candidate_rgba.png": rgba,
            "mask_overlay.png": geometry._mask_overlay(source, core, band),
            "difference.png": geometry._difference(source, candidate),
        }
        for filename, value in images.items():
            geometry._write_image(staging / filename, value)
        contact = geometry.make_contact_sheet((
            ("approved source", source),
            ("raw FLUX.2", raw),
            ("masked color candidate", candidate),
            ("semantic mask", images["mask_overlay.png"]),
            ("4x difference", images["difference.png"]),
            ("candidate RGBA", rgba),
        ))
        geometry._write_image(staging / "contact_sheet.png", contact)
        artifacts = {
            filename: _record(staging / filename, public_path=destination / filename)
            for filename in sorted((*images, "contact_sheet.png"))
        }
        manifest = {
            "schema": CANDIDATE_SCHEMA,
            "case_id": case_id,
            "base_asset_id": case["base_asset_id"],
            "attribute": case["attribute"],
            "target_color_name": case["target_color_name"],
            "target_color_srgb": case["target_color_srgb"],
            "state_classification": "research_candidate",
            "generation_status": "generated_pending_agent_2d_qa",
            "execution_policy": case["execution_policy"],
            "formal_registration_authorized": False,
            "user_acceptance": "not_claimed",
            "created_at_utc": _utc_now(),
            "jobs_contract": _record(OUTPUT_ROOT / "color_jobs_v3.json"),
            "runner": _record(RUNNER_PATH),
            "model": {
                "name": "black-forest-labs/FLUX.2-klein-4B",
                "revision": geometry.MODEL_REVISION,
                "inventory": _record(geometry.MODEL_INVENTORY),
                "local_files_only": True,
            },
            "source": case["source"],
            "mask_manifest": case["mask_manifest"],
            "mask_assets": case["mask_assets"],
            "parameters": {
                "prompt": case["prompt"],
                "negative_prompt": case["negative_prompt"],
                "seed": case["seed"],
                "width": geometry.WIDTH,
                "height": geometry.HEIGHT,
                "steps": geometry.STEPS,
                "guidance_scale": geometry.GUIDANCE_SCALE,
                "max_sequence_length": geometry.MAX_SEQUENCE_LENGTH,
                "physical_gpu": gpu,
            },
            "metrics": metrics,
            "automatic_2d_gate": "passed" if metrics["passed"] else "rejected",
            "artifacts": artifacts,
        }
        (staging / "candidate_manifest.json").write_bytes(geometry._json_bytes(manifest))
        geometry._readonly_tree(staging)
        geometry._fsync_tree(staging)
        geometry._rename_noreplace(staging, destination)
        return {"case_id": case_id, "status": "generated" if metrics["passed"] else "automatic_2d_rejected"}
    except BaseException as error:
        evidence = _publish_failure(case_id, staging, error) if staging.exists() else None
        if not isinstance(error, Exception):
            raise
        return {
            "case_id": case_id,
            "status": "generation_failure_rejected",
            "evidence": str(evidence) if evidence else None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def generate(case_ids: Sequence[str], gpu: str) -> list[dict[str, Any]]:
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ColorReferenceError("--case-id must be nonempty and unique")
    if unknown := sorted(set(case_ids) - set(CASE_BY_ID)):
        raise ColorReferenceError(f"unknown cases: {unknown}")
    contract = load_contract()
    pipeline = geometry._pipeline(gpu)
    return [generate_case(contract, case_id, pipeline, gpu) for case_id in case_ids]


def _load_candidate(case_id: str) -> tuple[Path, dict[str, Any]]:
    root = OUTPUT_ROOT / "cases" / case_id
    manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != CANDIDATE_SCHEMA
        or manifest.get("case_id") != case_id
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("runner") != _record(RUNNER_PATH)
    ):
        raise ColorReferenceError("color candidate schema/state/runner changed")
    for filename, value in manifest.get("artifacts", {}).items():
        if value != _record(root / filename, mode=0o444):
            raise ColorReferenceError(f"color artifact changed: {case_id}/{filename}")
    return root, manifest


def review(case_id: str, status: str, notes: str) -> Path:
    if case_id not in CASE_BY_ID or status not in {"agent_2d_passed", "rejected"}:
        raise ColorReferenceError("unknown case or decision")
    if not notes.strip():
        raise ColorReferenceError("review notes must be nonempty")
    root, manifest = _load_candidate(case_id)
    destination = root / "agent_2d_visual_qa.json"
    passed = status == "agent_2d_passed"
    if passed and manifest.get("automatic_2d_gate") != "passed":
        raise ColorReferenceError("agent cannot pass automatic color rejection")
    payload = {
        "schema": DECISION_SCHEMA,
        "case_id": case_id,
        "status": status,
        "state_classification": "research_candidate" if passed else "rejected",
        "reviewer_kind": "agent",
        "reviewer": "codex_female_route2_base",
        "checks": {
            "only_target_color_changed": passed,
            "identity_face_pose_camera_preserved": passed,
            "target_geometry_and_texture_preserved": passed,
            "non_target_regions_preserved": passed,
            "pixal_bindable_silhouette_unchanged": passed,
        },
        "notes": notes.strip(),
        "candidate_manifest": _record(root / "candidate_manifest.json", mode=0o444),
        "contact_sheet": manifest["artifacts"]["contact_sheet.png"],
        "pixal_authorized": passed,
        "formal_dataset_registration_authorized": False,
        "user_acceptance": "not_claimed",
        "reviewed_at_utc": _utc_now(),
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        os.write(descriptor, geometry._json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def finalize() -> Path:
    destination = OUTPUT_ROOT / "review_summary_v1"
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    records, panels, jobs = [], [], []
    for case_id, case in CASE_BY_ID.items():
        root, candidate = _load_candidate(case_id)
        decision = json.loads((root / "agent_2d_visual_qa.json").read_text(encoding="utf-8"))
        if decision.get("schema") != DECISION_SCHEMA or decision.get("case_id") != case_id:
            raise ColorReferenceError(f"color decision changed: {case_id}")
        with Image.open(root / "contact_sheet.png") as opened:
            panels.append((case_id, opened.convert("RGB")))
        records.append({
            "case_id": case_id,
            "candidate_manifest": _record(root / "candidate_manifest.json", mode=0o444),
            "decision": _record(root / "agent_2d_visual_qa.json", mode=0o444),
            "status": decision["status"],
        })
        if decision["status"] == "agent_2d_passed":
            jobs.append({
                "asset_id": f"route2_color_v3_{case_id}",
                "base_asset_id": candidate["base_asset_id"],
                "attribute": candidate["attribute"],
                "target_color_name": candidate["target_color_name"],
                "state_classification": "research_candidate",
                "input_rgba": candidate["artifacts"]["candidate_rgba.png"],
                "reference_manifest": _record(root / "candidate_manifest.json", mode=0o444),
                "reference_decision": _record(root / "agent_2d_visual_qa.json", mode=0o444),
                "model": {"name": "TencentARC/Pixal3D", "revision": geometry.PIXAL_REVISION},
                "parameters": {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True},
                "output_dir": str(PIXAL_OUTPUT_ROOT / f"route2_color_v3_{case_id}"),
                "execution_status": "ready_for_pixal_color_feasibility_preflight",
            })
    staging = Path(tempfile.mkdtemp(prefix=".review_summary_v1.", suffix=".staging", dir=OUTPUT_ROOT))
    try:
        geometry._write_image(staging / "all_cases_contact_sheet.png", geometry.make_contact_sheet(panels))
        (staging / "pixal_jobs_v1.json").write_bytes(geometry._json_bytes({
            "schema": PIXAL_JOBS_SCHEMA,
            "state_classification": "research_candidate_preflight",
            "formal_registration_authorized": False,
            "ordinary_production_color_policy": "deterministic_semantic_material_transform",
            "source_jobs_contract": _record(OUTPUT_ROOT / "color_jobs_v3.json", mode=0o444),
            "jobs": jobs,
        }))
        (staging / "summary.json").write_bytes(geometry._json_bytes({
            "schema": "route2_controlled_color_reference_summary_v1",
            "state_classification": "research_candidate_preflight",
            "formal_registration_authorized": False,
            "case_count": len(records),
            "passed_count": sum(item["status"] == "agent_2d_passed" for item in records),
            "rejected_count": sum(item["status"] == "rejected" for item in records),
            "cases": records,
            "pixal_job_count": len(jobs),
            "created_at_utc": _utc_now(),
        }))
        geometry._readonly_tree(staging)
        geometry._fsync_tree(staging)
        geometry._rename_noreplace(staging, destination)
        return destination / "summary.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--case-id", action="append", required=True)
    generate_parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--case-id", choices=tuple(CASE_BY_ID), required=True)
    review_parser.add_argument("--status", choices=("agent_2d_passed", "rejected"), required=True)
    review_parser.add_argument("--notes", required=True)
    commands.add_parser("finalize")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        print(f"ROUTE2_CONTROLLED_COLOR_PREPARED {prepare()}")
    elif args.command == "generate":
        print(json.dumps(generate(args.case_id, args.gpu), indent=2, sort_keys=True))
    elif args.command == "review":
        print(f"ROUTE2_CONTROLLED_COLOR_REVIEWED {review(args.case_id, args.status, args.notes)}")
    elif args.command == "finalize":
        print(f"ROUTE2_CONTROLLED_COLOR_FINALIZED {finalize()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
