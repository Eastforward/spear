from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from PIL import Image
import pytest

from tools import adopt_direct_animal_pixal_attempt as adopter
from tools import controlled_animal_isnet_worker as isnet
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import run_controlled_animal_static_reviews as static_reviews


INSTANCE_ID = "dog_shiba_inu_direct_fixture_v1"
SEED = 41


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _record(path: Path) -> dict:
    path = path.absolute()
    return {
        "path": str(path),
        "sha256": adopter._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _encoded_png(color: tuple[int, int, int]) -> bytes:
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(encoded, format="PNG")
    return encoded.getvalue()


def _write_glb(path: Path, *, variant: str = "valid") -> None:
    binary = bytearray()
    buffer_views: list[dict] = []

    def append_buffer_view(payload: bytes) -> int:
        binary.extend(b"\x00" * ((-len(binary)) % 4))
        index = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
        )
        binary.extend(payload)
        return index

    position_view = append_buffer_view(
        struct.pack(
            "<9f",
            -0.5,
            0.0,
            -0.5,
            0.5,
            0.0,
            -0.5,
            0.0,
            1.0,
            0.5,
        )
    )
    texcoord_view = append_buffer_view(
        struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.5, 1.0)
    )
    packed_images = [
        b"\x00" if variant == "fake_png" else _encoded_png((20, 40, 60)),
        _encoded_png((80, 100, 120)),
    ]
    image_views = [append_buffer_view(payload) for payload in packed_images]
    material = (
        {}
        if variant == "empty_material"
        else {
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicRoughnessTexture": {"index": 1},
            }
        }
    )
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [-0.5, 0.0, -0.5],
                "max": [0.5, 1.0, 0.5],
            },
            {
                "bufferView": texcoord_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [material],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [
            {"bufferView": image_views[0], "mimeType": "image/png"},
            {"bufferView": image_views[1], "mimeType": "image/png"},
        ],
        "skins": [],
        "animations": [],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _target(*, complete: bool = True) -> dict:
    if not complete:
        return {
            "measurement": "shoulder_height_cm",
            "reference_value_cm": 39,
            "tolerance_cm": 4,
            "status": "provisional_research_target",
        }
    return {
        "control_attribute": "size",
        "measurement": "shoulder_height_cm",
        "mode": "relative_to_profile_reference",
        "profile_id": "dog_shiba_inu_physical_candidate_v1",
        "reference_provenance": {
            "artifact": None,
            "notes": "fixture",
            "source_id": "fixture",
            "status": "provisional",
        },
        "reference_value_cm": 39,
        "scale_ratio": 1,
        "selected_value": "medium",
        "target_value_cm": 39,
        "tolerance_cm": 4,
    }


def _rig() -> dict:
    return {
        "actions": ["Walking", "Idle"],
        "front_axis": "positive_x",
        "profile_id": "quadruped_dog_v1",
        "skeleton_family": "quaternius_dog",
    }


def _imagegen_fixture(
    root: Path,
    *,
    provider_mode: str = "codex_builtin_image_gen",
    scheduling_mode: str = "fixed_partition_v1",
    job_seed: int = SEED,
    complete_target: bool = True,
    glb_variant: str = "valid",
) -> tuple[Path, dict]:
    source = root / "source"
    candidate = source / "candidate.png"
    alpha = source / "alpha.png"
    rgba = source / "input_rgba.png"
    source.mkdir(parents=True)
    Image.new("RGB", (16, 16), (170, 90, 50)).save(candidate)
    alpha_image = Image.new("L", (16, 16), 0)
    alpha_image.paste(255, (2, 3, 14, 15))
    alpha_image.save(alpha)
    rgba_image = Image.new("RGBA", (16, 16), (170, 90, 50, 0))
    rgba_image.putalpha(alpha_image)
    rgba_image.save(rgba)

    receipt = {
        "schema": "avengine_builtin_imagegen_reference_generation_receipt_v3",
        "created_date": "2026-07-28",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "tool": {
            "mode": provider_mode,
            "provider_model_revision": None,
            "seed": None,
            "replayability": "output_hash_bound_only",
            "invocation_ordinal": 0,
            "invocations_for_method": 1,
            "images_for_method": 1,
        },
        "method_revision": {
            "kind": "new_four_limb_image_method_revision",
            "this_is_not_a_pixel3d_seed_retry": True,
            "pixel3d_seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
        },
        "prompt": {"fixture": True},
        "output": {
            **_record(candidate),
            "canvas": [16, 16],
            "mode": "RGB",
        },
    }
    receipt_path = source / "generation_receipt.json"
    _write_json(receipt_path, receipt)
    review = {
        "schema": "avengine_research_animal_image_2d_review_v1",
        "candidate": _record(candidate),
        "generation_receipt": _record(receipt_path),
        "segmentation": {
            "alpha_path": str(alpha.absolute()),
            "alpha_sha256": adopter._sha256_file(alpha),
            "rgba_path": str(rgba.absolute()),
            "rgba_sha256": adopter._sha256_file(rgba),
            "foreground_bbox_xyxy": [2, 3, 13, 14],
            "threshold": 128,
        },
        "decision": "approved_for_dynamic_feasibility_only",
        "downstream_gate": "one_hash_bound_new_pixel3d_checkpoint_authorized",
        "constraints": {
            "pixel3d_invocations_allowed": 1,
            "pixel3d_seed_retry_allowed": False,
            "pixel3d_candidate_ranking_allowed": False,
            "formal_dataset_registration_authorized": False,
        },
    }
    review_path = source / "objective_2d_review.json"
    _write_json(review_path, review)
    isnet_jobs = {
        "schema": isnet.JOBS_SCHEMA,
        "jobs": [
            {
                "instance_id": INSTANCE_ID,
                "candidate_path": str(candidate.absolute()),
                "candidate_sha256": adopter._sha256_file(candidate),
                "alpha_path": str(alpha.absolute()),
                "rgba_path": str(rgba.absolute()),
            }
        ],
    }
    isnet_jobs_path = source / "isnet_jobs.json"
    _write_json(isnet_jobs_path, isnet_jobs)
    isnet_status = {
        "schema": isnet.STATUS_SCHEMA,
        "status": "passed",
        "passed_count": 1,
        "failed_count": 0,
        "model": {
            "name": "isnet-general-use",
            "path": str(isnet.MODEL_PATH),
            "sha256": isnet.MODEL_SHA256,
        },
        "model_load_seconds": 1.0,
        "jobs": [
            {
                "instance_id": INSTANCE_ID,
                "status": "passed",
                "alpha_extrema": [0, 255],
                "alpha_path": str(alpha.absolute()),
                "alpha_sha256": adopter._sha256_file(alpha),
                "rgba_path": str(rgba.absolute()),
                "rgba_sha256": adopter._sha256_file(rgba),
                "foreground_bbox_xyxy": [2, 3, 14, 15],
                "foreground_fraction_at_128": 0.5625,
                "wall_seconds": 0.1,
            }
        ],
    }
    isnet_status_path = source / "isnet_status.json"
    _write_json(isnet_status_path, isnet_status)

    output = root / "raw" / INSTANCE_ID / "pixal_raw_1024.glb"
    manifest_path = output.with_suffix(".manifest.json")
    _write_glb(output, variant=glb_variant)
    controlled = {
        "schema": "avengine_research_animal_pixal_request_v1",
        "asset_class": "animal",
        "execution_job_id": "research_animal_fixture",
        "generation_seed": SEED,
        "instance_id": INSTANCE_ID,
        "profile_schema_id": "dog_shiba_inu_imagegen_research_fixture_v1",
        "method_freeze_sha256": adopter._sha256_file(receipt_path),
        "rig_profile": _rig(),
        "route": adopter.IMAGEGEN_ROUTE,
        "sampled_attributes": {
            "body_build": "standard",
            "coat_color": "red_white",
            "size": "medium",
        },
        "target_physical_profile": _target(complete=complete_target),
        "formal_dataset_registration_authorized": False,
    }
    controlled["request_sha256"] = adopter._hash_without(
        controlled, "request_sha256"
    )
    job = {
        "legacy_tag": INSTANCE_ID,
        "candidate_tag": f"{INSTANCE_ID}_pixal_dynamic_v1",
        "seed": job_seed,
        "attempt_ordinal": 0,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "reference": {
            "source": _record(candidate),
            "pixal_input": _record(rgba),
            "normalization": "pinned_isnet_general_use_alpha_v1",
        },
        "output": str(output.absolute()),
        "manifest": str(manifest_path.absolute()),
        "public_output": str(output.absolute()),
        "public_manifest": str(manifest_path.absolute()),
        "controlled_request": controlled,
        "rig_mode": "animated_transfer",
    }
    jobs_path = root / "inputs" / "pixal_jobs.json"
    _write_json(jobs_path, [job])
    manifest = {
        "backend": "pixal3d",
        "controlled_request": controlled,
        "dino": {
            "revision": pixal_inputs.DINO_REVISION,
            "snapshot": "/models/dino",
        },
        "input": {
            "alpha_max": 255,
            "alpha_min": 0,
            "mode": "RGBA",
            "path": str(rgba.absolute()),
            "sha256": adopter._sha256_file(rgba),
            "size": [16, 16],
        },
        "model": {
            "revision": pixal_inputs.PIXAL_MODEL_REVISION,
            "snapshot": "/models/pixal",
        },
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "output": {
            "bytes": output.stat().st_size,
            "path": str(output.absolute()),
            "sha256": adopter._sha256_file(output),
        },
        "parameters": {
            "low_vram": False,
            "manual_fov": 0.2,
            "resolution": 1024,
            "seed": job_seed,
        },
        "timings": {
            "inference_and_export_seconds": 1.0,
            "model_reused": True,
            "persistent_worker_model_load_seconds": 2.0,
        },
    }
    _write_json(manifest_path, manifest)
    worker = {
        "failed_count": 0,
        "finished_at": "2026-07-28T00:00:03+00:00",
        "gpu": 2,
        "jobs": [
            {
                "attempt_ordinal": 0,
                "candidate_tag": job["candidate_tag"],
                "finished_at": "2026-07-28T00:00:03+00:00",
                "legacy_tag": INSTANCE_ID,
                "manifest": str(manifest_path.absolute()),
                "output": str(output.absolute()),
                "output_sha256": adopter._sha256_file(output),
                "seed": job_seed,
                "started_at": "2026-07-28T00:00:02+00:00",
                "status": "passed",
                "wall_seconds": 1.0,
            }
        ],
        "low_vram": False,
        "model_load_seconds": 2.0,
        "passed_count": 1,
        "scheduling_mode": scheduling_mode,
        "schema": "pixal_animal_persistent_worker_v1",
        "started_at": "2026-07-28T00:00:00+00:00",
    }
    worker_path = root / "raw" / "pixal_worker_status.json"
    _write_json(worker_path, worker)
    evidence_paths = {
        "generation_receipt": receipt_path,
        "objective_2d_review": review_path,
        "isnet_jobs": isnet_jobs_path,
        "isnet_status": isnet_status_path,
        "pixal_jobs": jobs_path,
        "pixal_worker_status": worker_path,
        "pixal_attempt_manifest": manifest_path,
        "pixal_raw_glb": output,
    }
    spec = {
        "schema": adopter.SPEC_SCHEMA,
        "route": adopter.IMAGEGEN_ROUTE,
        "source_kind": adopter.IMAGEGEN_SOURCE_KIND,
        "instance_id": INSTANCE_ID,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "evidence": {
            role: _record(path) for role, path in evidence_paths.items()
        },
    }
    spec["spec_sha256"] = adopter._hash_without(spec, "spec_sha256")
    spec_path = root / "adoption_spec.json"
    _write_json(spec_path, spec)
    return spec_path, spec


def _british_fixture(root: Path, *, execution_seed: int) -> tuple[Path, dict]:
    instance_id = "cat_british_shorthair_shadow_cleanup_v1"
    candidate = root / "lineage" / "candidate.png"
    original_alpha = root / "lineage" / "alpha.png"
    original_rgba = root / "lineage" / "input_rgba.png"
    cleaned_alpha = root / "cleanup" / "alpha_cleaned.png"
    cleaned_rgba = root / "cleanup" / "input_rgba_cleaned.png"
    removed_mask = root / "cleanup" / "removed.png"
    candidate.parent.mkdir(parents=True)
    cleaned_alpha.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (80, 100, 120)).save(candidate)
    before = Image.new("L", (64, 64), 255)
    before.save(original_alpha)
    before_rgba = Image.new("RGBA", (64, 64), (80, 100, 120, 255))
    before_rgba.save(original_rgba)
    mask_bytes = bytearray(64 * 64)
    mask_bytes[:1223] = b"\xff" * 1223
    mask = Image.frombytes("L", (64, 64), bytes(mask_bytes))
    mask.save(removed_mask)
    after_bytes = bytearray(b"\xff" * (64 * 64))
    after_bytes[:1223] = b"\x00" * 1223
    after = Image.frombytes("L", (64, 64), bytes(after_bytes))
    after.save(cleaned_alpha)
    after_rgba = Image.new("RGBA", (64, 64), (80, 100, 120, 255))
    after_rgba.putalpha(after)
    after_rgba.save(cleaned_rgba)

    cleanup = {
        "schema": "avengine_generated_animal_opaque_ground_shadow_cleanup_v2",
        "state_classification": "technical_spike_input_repair_only",
        "formal_dataset_registration_authorized": False,
        "execution": {
            "class": "deterministic_cpu_only",
            "connectivity": 8,
            "coordinate_convention": "xyxy_half_open",
            "gpu_or_pixal_started": False,
        },
        "decision": {
            "abdomen_and_leg_evidence_preserved": True,
            "dynamic_asset_admission_authorized": False,
            "next_step": "fixture",
            "opaque_ground_shadow_cleanup_passed": True,
            "paw_bottom_evidence_preserved": True,
            "pixal_seed_unchanged_for_next_run": True,
            "source_candidate_unchanged": True,
        },
        "measurements": {
            "selected_shadow_pixel_count": 1223,
            "removed_alpha_pixel_count": 1223,
            "paw_bottom_evidence": [
                {"byte_exact_preserved": True} for _index in range(4)
            ],
            "protected_body_evidence": [
                {"byte_exact_preserved": True} for _index in range(2)
            ],
        },
        "source": {
            "candidate": {**_record(candidate), "mode": "RGB"},
            "input_alpha": {**_record(original_alpha), "mode": "L"},
            "input_rgba": {**_record(original_rgba), "mode": "RGBA"},
            "pixal_seed_decimal_frozen_for_controlled_rerun": (
                adopter.BRITISH_SEED_DECIMAL_ATTESTATION
            ),
        },
        "outputs": {
            "cleaned_alpha": {**_record(cleaned_alpha), "mode": "L"},
            "cleaned_rgba": {**_record(cleaned_rgba), "mode": "RGBA"},
            "removed_shadow_mask": {**_record(removed_mask), "mode": "L"},
        },
    }
    cleanup_path = root / "cleanup" / "shadow_cleanup_manifest.json"
    _write_json(cleanup_path, cleanup)

    predecessor_raw = root / "predecessor" / "pixal_raw_1024.glb"
    _write_glb(predecessor_raw)
    predecessor_manifest = predecessor_raw.with_suffix(".manifest.json")
    predecessor = {
        "backend": "pixal3d",
        "controlled_request": {
            "generation_seed": 3750817048556488970,
        },
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "output": {
            "bytes": predecessor_raw.stat().st_size,
            "path": str(predecessor_raw.absolute()),
            "sha256": adopter._sha256_file(predecessor_raw),
        },
        "parameters": {"seed": 3750817048556488970},
    }
    _write_json(predecessor_manifest, predecessor)
    owner_review = root / "predecessor" / "owner_review.json"
    static_decision = root / "predecessor" / "static_decision.json"
    _write_json(owner_review, {"decision": "approved_for_pixal3d"})
    _write_json(static_decision, {"state": "rejected"})
    rejection = {
        "schema": "avengine_pixal_same_mesh_mirrored_limb_repair_v1",
        "formal_dataset_registration_authorized": False,
        "decision": {
            "status": "rejected_bounded_local_repair_preflight",
            "cat_semantic_retarget_authorized": False,
        },
        "lineage": {
            "approved_reference": _record(candidate),
            "owner_review": _record(owner_review),
            "pixal_manifest": _record(predecessor_manifest),
            "pixal_source": _record(predecessor_raw),
            "static_decision": _record(static_decision),
        },
    }
    rejection_path = root / "predecessor" / "rejected_manifest.json"
    _write_json(rejection_path, rejection)

    request = {
        "asset_class": "animal",
        "execution_job_id": "research_animal_input_repair_fixture",
        "formal_dataset_registration_authorized": False,
        "generation_seed": execution_seed,
        "instance_id": instance_id,
        "method_revision": {
            "candidate_ranking_allowed": False,
            "input_repair_kind": "bounded_ground_shadow_alpha_cleanup_v1",
            "pixal_seed_unchanged": True,
            "same_source_candidate": True,
            "seed_retry_allowed": False,
            "this_is_not_a_seed_retry": True,
        },
        "profile_schema_id": "cat_british_shorthair_shadow_cleanup_fixture_v1",
        "reference": {
            "cleaned_pixal_input": _record(cleaned_rgba),
            "input_repair_manifest": _record(cleanup_path),
            "original_pixal_input": _record(original_rgba),
            "source": _record(candidate),
        },
        "rejected_predecessor": {
            "bounded_repair_rejection": _record(rejection_path),
            "pixal_manifest": _record(predecessor_manifest),
            "pixal_raw": _record(predecessor_raw),
        },
        "rig_profile": {
            "actions": ["Walking", "Idle"],
            "front_axis": "positive_x",
            "profile_id": "quadruped_cat_v1",
            "skeleton_family": "quaternius_cat",
        },
        "route": adopter.SHADOW_CLEANUP_ROUTE,
        "sampled_attributes": {
            "body_build": "stocky",
            "coat_color": "blue",
            "size": "medium",
        },
        "schema": "avengine_research_animal_pixal_input_repair_request_v1",
        "target_physical_profile": {
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
            "mode": "relative_to_profile_reference",
            "profile_id": "cat_british_shorthair_physical_candidate_v1",
            "reference_value_cm": 30,
            "scale_ratio": 1,
            "selected_value": "medium",
            "target_value_cm": 30.0,
            "tolerance_cm": 3,
        },
    }
    request_path = root / "inputs" / "controlled_request.json"
    _write_json(request_path, request)
    controlled = {
        **copy.deepcopy(request),
        "request_record": _record(request_path),
        "request_sha256": adopter._sha256_file(request_path),
    }
    output = root / "raw" / instance_id / "pixal_raw_1024.glb"
    _write_glb(output)
    manifest_path = output.with_suffix(".manifest.json")
    job = {
        "attempt_ordinal": 0,
        "candidate_tag": f"{instance_id}_pixal_dynamic_v1",
        "controlled_request": controlled,
        "legacy_tag": instance_id,
        "manifest": str(manifest_path.absolute()),
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "output": str(output.absolute()),
        "public_manifest": str(manifest_path.absolute()),
        "public_output": str(output.absolute()),
        "reference": {
            "input_repair": {
                "manifest_path": str(cleanup_path.absolute()),
                "manifest_sha256": adopter._sha256_file(cleanup_path),
                "manifest_size_bytes": cleanup_path.stat().st_size,
                "method": "bounded_ground_shadow_alpha_cleanup_v1",
            },
            "normalization": (
                "pinned_isnet_general_use_alpha_plus_bounded_shadow_cleanup_v1"
            ),
            "pixal_input": _record(cleaned_rgba),
            "source": _record(candidate),
        },
        "seed": execution_seed,
    }
    jobs_path = root / "inputs" / "pixal_jobs.json"
    _write_json(jobs_path, [job])
    manifest = {
        "backend": "pixal3d",
        "controlled_request": controlled,
        "dino": {
            "revision": pixal_inputs.DINO_REVISION,
            "snapshot": "/models/dino",
        },
        "input": {
            "alpha_max": 255,
            "alpha_min": 0,
            "mode": "RGBA",
            "path": str(cleaned_rgba.absolute()),
            "sha256": adopter._sha256_file(cleaned_rgba),
            "size": [64, 64],
        },
        "model": {
            "revision": pixal_inputs.PIXAL_MODEL_REVISION,
            "snapshot": "/models/pixal",
        },
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "output": {
            "bytes": output.stat().st_size,
            "path": str(output.absolute()),
            "sha256": adopter._sha256_file(output),
        },
        "parameters": {
            "low_vram": False,
            "manual_fov": 0.2,
            "resolution": 1024,
            "seed": execution_seed,
        },
        "timings": {
            "inference_and_export_seconds": 1.0,
            "model_reused": True,
            "persistent_worker_model_load_seconds": 2.0,
        },
    }
    _write_json(manifest_path, manifest)
    worker = {
        "failed_count": 0,
        "finished_at": "2026-07-28T00:00:03+00:00",
        "gpu": 3,
        "jobs": [
            {
                "attempt_ordinal": 0,
                "candidate_tag": job["candidate_tag"],
                "finished_at": "2026-07-28T00:00:03+00:00",
                "legacy_tag": instance_id,
                "manifest": str(manifest_path.absolute()),
                "output": str(output.absolute()),
                "output_sha256": adopter._sha256_file(output),
                "seed": execution_seed,
                "started_at": "2026-07-28T00:00:02+00:00",
                "status": "passed",
                "wall_seconds": 1.0,
            }
        ],
        "low_vram": False,
        "model_load_seconds": 2.0,
        "passed_count": 1,
        "scheduling_mode": "fixed_partition_v1",
        "schema": "pixal_animal_persistent_worker_v1",
        "started_at": "2026-07-28T00:00:00+00:00",
    }
    worker_path = root / "raw" / "pixal_worker_status.json"
    _write_json(worker_path, worker)
    evidence_paths = {
        "controlled_request": request_path,
        "shadow_cleanup_manifest": cleanup_path,
        "pixal_jobs": jobs_path,
        "pixal_worker_status": worker_path,
        "pixal_attempt_manifest": manifest_path,
        "pixal_raw_glb": output,
    }
    spec = {
        "schema": adopter.SPEC_SCHEMA,
        "route": adopter.SHADOW_CLEANUP_ROUTE,
        "source_kind": adopter.SHADOW_CLEANUP_SOURCE_KIND,
        "instance_id": instance_id,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "evidence": {
            role: _record(path) for role, path in evidence_paths.items()
        },
    }
    spec["spec_sha256"] = adopter._hash_without(spec, "spec_sha256")
    spec_path = root / "adoption_spec.json"
    _write_json(spec_path, spec)
    return spec_path, spec


def test_imagegen_adoption_is_create_only_and_static_review_dispatches(tmp_path):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    output_root = tmp_path / "adopted"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(adopter.__file__).resolve()),
            "--spec",
            str(spec_path),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DIRECT_ANIMAL_PIXAL_ADOPTION_OK" in completed.stdout
    batch_path = output_root / "pixal_batch_manifest.json"

    _, batch = adopter.load_adopted_batch(batch_path)
    _, dispatched = static_reviews.load_pixal_batch(batch_path)
    assert dispatched == batch
    assert batch["formal_dataset_registration_authorized"] is False
    assert batch["execution_evidence"]["inference_performed_by_adopter"] is False
    assert batch["execution_evidence"]["scheduling_mode"] == "fixed_partition_v1"
    assert batch["reference_label"] == "hash-bound ImageGen input"
    assert (
        static_reviews._reference_label_for_batch(batch)
        == "hash-bound ImageGen input"
    )
    assert (
        batch["attempts"][0]["output"]["sha256"]
        == batch["copied_artifacts"]["pixal_raw_glb"]["sha256"]
    )
    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        adopter.adopt_attempt(spec_path, output_root)


def _shiba_direct_semantics():
    return {
        "taxonomy": {"species": "dog", "breed": "shiba_inu"},
        "fixed_attributes": {
            "life_stage": "adult",
            "coat_length": "short",
            "coat_pattern": "urajiro",
            "ear_shape": "upright",
            "tail_shape": "curled",
        },
        "lineage_group_id": "dog_shiba_inu_direct_pixel3d_v1",
        "acoustic_profile": {
            "profile_id": "dog_vocalization_v1",
            "default_event_class": "dog_bark",
            "allowed_event_classes": ["dog_bark", "dog_growl", "silent"],
            "selection_attributes": ["species", "breed", "life_stage"],
        },
    }


def test_direct_source_authority_reauthenticates_adopted_batch(tmp_path):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    batch_path = adopter.adopt_attempt(spec_path, tmp_path / "adopted")
    authority = adopter.build_direct_source_authority(
        batch_path,
        **_shiba_direct_semantics(),
    )
    authority_path = tmp_path / "source_authority.json"
    _write_json(authority_path, authority)

    loaded_path, loaded, context = adopter.load_direct_source_authority(
        authority_path,
        expected_sha256=adopter._sha256_file(authority_path),
    )

    assert loaded_path == authority_path.resolve()
    assert loaded == authority
    assert loaded["state_classification"] == "research_candidate"
    assert loaded["formal_dataset_registration_authorized"] is False
    assert context["adopted_batch_path"] == batch_path.resolve()
    assert context["attempt"]["instance_id"] == INSTANCE_ID
    assert context["adoption_context"]["controlled"]["rig_profile"] == _rig()


def test_cli_writes_canonical_direct_source_authority_after_adoption(
    tmp_path,
    capsys,
):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    semantics_path = tmp_path / "source_authority_semantics.json"
    _write_json(semantics_path, _shiba_direct_semantics())
    output_root = tmp_path / "adopted"
    authority_path = tmp_path / "authority" / "source_authority.json"

    assert adopter.main(
        [
            "--spec",
            str(spec_path),
            "--output-root",
            str(output_root),
            "--source-authority-output",
            str(authority_path),
            "--source-authority-semantics",
            str(semantics_path),
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert "DIRECT_ANIMAL_PIXAL_ADOPTION_OK" in stdout
    assert "DIRECT_ANIMAL_SOURCE_AUTHORITY_OK" in stdout
    loaded_path, authority, context = adopter.load_direct_source_authority(
        authority_path,
        expected_sha256=adopter._sha256_file(authority_path),
    )
    assert loaded_path == authority_path.resolve()
    assert authority["instance_id"] == INSTANCE_ID
    assert context["adopted_batch_path"] == (
        output_root / "pixal_batch_manifest.json"
    ).resolve()


def test_cli_refuses_to_replace_existing_source_authority(
    tmp_path,
    capsys,
):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    semantics_path = tmp_path / "source_authority_semantics.json"
    _write_json(semantics_path, _shiba_direct_semantics())
    output_root = tmp_path / "adopted"
    authority_path = tmp_path / "authority" / "source_authority.json"
    authority_path.parent.mkdir(parents=True)
    original = b"existing authority must remain unchanged\n"
    authority_path.write_bytes(original)

    assert adopter.main(
        [
            "--spec",
            str(spec_path),
            "--output-root",
            str(output_root),
            "--source-authority-output",
            str(authority_path),
            "--source-authority-semantics",
            str(semantics_path),
        ]
    ) == 2

    assert authority_path.read_bytes() == original
    assert (output_root / "pixal_batch_manifest.json").is_file()
    assert "refusing to replace" in capsys.readouterr().err
    assert not list(
        authority_path.parent.glob(f".{authority_path.name}.*.staging")
    )


def test_cli_default_does_not_build_or_write_source_authority(
    tmp_path,
    monkeypatch,
    capsys,
):
    output_root = tmp_path / "adopted"
    manifest = output_root / "pixal_batch_manifest.json"

    def adopt_attempt(_spec_path, requested_output_root):
        assert requested_output_root == output_root
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    def unexpected_authority_build(*_args, **_kwargs):
        raise AssertionError("default adoption must not build source authority")

    monkeypatch.setattr(adopter, "adopt_attempt", adopt_attempt)
    monkeypatch.setattr(
        adopter,
        "build_direct_source_authority",
        unexpected_authority_build,
    )

    assert adopter.main(
        [
            "--spec",
            str(tmp_path / "unused_spec.json"),
            "--output-root",
            str(output_root),
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == (
        f"DIRECT_ANIMAL_PIXAL_ADOPTION_OK output={manifest}"
    )
    assert not list(tmp_path.rglob("source_authority.json"))


def test_direct_source_authority_rejects_semantic_profile_reseal(tmp_path):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    batch_path = adopter.adopt_attempt(spec_path, tmp_path / "adopted")
    authority = adopter.build_direct_source_authority(
        batch_path,
        **_shiba_direct_semantics(),
    )
    authority["taxonomy"]["breed"] = "another_breed"
    authority["authority_sha256"] = adopter._hash_without(
        authority, "authority_sha256"
    )
    authority_path = tmp_path / "resealed_authority.json"
    _write_json(authority_path, authority)

    with pytest.raises(contracts.ContractError, match="semantic profile hash"):
        adopter.load_direct_source_authority(authority_path)


def test_british_cleanup_accepts_only_exact_frozen_seed(tmp_path):
    spec_path, _spec = _british_fixture(
        tmp_path / "british",
        execution_seed=adopter.BRITISH_EXECUTION_SEED,
    )
    _, spec, context = adopter.load_adoption_spec(spec_path)

    assert spec["route"] == adopter.SHADOW_CLEANUP_ROUTE
    assert context["seed"] == adopter.BRITISH_EXECUTION_SEED
    assert (
        adopter.ROUTE_CONTRACTS[spec["route"]]["reference_label"]
        == "bounded shadow-cleaned input"
    )
    with pytest.raises(contracts.ContractError, match="frozen British seed"):
        adopter._validate_british_seed(3750817048556489000, "rounded seed")


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"provider_mode": "unknown_image_provider"}, "unknown or replayable"),
        ({"scheduling_mode": "shared_claim_queue_v1"}, "fixed partition"),
        ({"job_seed": SEED + 1}, "seed/attempt identity"),
        ({"complete_target": False}, "target physical profile is incomplete"),
        ({"glb_variant": "fake_png"}, "deep PBR validation"),
        ({"glb_variant": "empty_material"}, "deep PBR validation"),
    ],
)
def test_imagegen_semantic_tampering_is_rejected(
    tmp_path, fixture_kwargs, message
):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source", **fixture_kwargs)

    with pytest.raises(contracts.ContractError, match=message):
        adopter.load_adoption_spec(spec_path)


def test_spec_hash_route_and_formal_tampering_are_rejected(tmp_path):
    spec_path, spec = _imagegen_fixture(tmp_path / "source")

    tampered_hash = copy.deepcopy(spec)
    tampered_hash["evidence"]["pixal_raw_glb"]["sha256"] = "0" * 64
    tampered_hash["spec_sha256"] = adopter._hash_without(
        tampered_hash, "spec_sha256"
    )
    tampered_hash_path = tmp_path / "tampered_hash.json"
    _write_json(tampered_hash_path, tampered_hash)
    with pytest.raises(contracts.ContractError, match="changed"):
        adopter.load_adoption_spec(tampered_hash_path)

    tampered_route = copy.deepcopy(spec)
    tampered_route["route"] = "unknown_direct_provider_route_v1"
    tampered_route["spec_sha256"] = adopter._hash_without(
        tampered_route, "spec_sha256"
    )
    tampered_route_path = tmp_path / "tampered_route.json"
    _write_json(tampered_route_path, tampered_route)
    with pytest.raises(contracts.ContractError, match="contract/hash"):
        adopter.load_adoption_spec(tampered_route_path)

    formal = copy.deepcopy(spec)
    formal["formal_dataset_registration_authorized"] = True
    formal["spec_sha256"] = adopter._hash_without(formal, "spec_sha256")
    formal_path = tmp_path / "formal.json"
    _write_json(formal_path, formal)
    with pytest.raises(contracts.ContractError, match="contract/hash"):
        adopter.load_adoption_spec(formal_path)


def test_noncanonical_instance_id_is_rejected_before_adoption(tmp_path):
    spec_path, spec = _imagegen_fixture(tmp_path / "source")
    tampered = copy.deepcopy(spec)
    tampered["instance_id"] = "../escaped"
    tampered["spec_sha256"] = adopter._hash_without(tampered, "spec_sha256")
    _write_json(spec_path, tampered)

    with pytest.raises(contracts.ContractError, match="canonical"):
        adopter.load_adoption_spec(spec_path)


def test_adopted_batch_rejects_unknown_schema_and_resealed_route(tmp_path):
    spec_path, _spec = _imagegen_fixture(tmp_path / "source")
    batch_path = adopter.adopt_attempt(spec_path, tmp_path / "adopted")
    batch = contracts.load_json(batch_path)

    unknown = copy.deepcopy(batch)
    unknown["schema"] = "unknown_pixal_batch_v1"
    unknown["batch_sha256"] = adopter._hash_without(unknown, "batch_sha256")
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(
        json.dumps(unknown, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(contracts.ContractError, match="unsupported Pixal batch"):
        static_reviews.load_pixal_batch(unknown_path)

    route = copy.deepcopy(batch)
    route["route"] = adopter.SHADOW_CLEANUP_ROUTE
    route["batch_sha256"] = adopter._hash_without(route, "batch_sha256")
    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(route, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(contracts.ContractError, match="contract/hash"):
        adopter.load_adopted_batch(route_path)

    tampered_root = tmp_path / "tampered_adopted"
    shutil.copytree(batch_path.parent, tampered_root)
    tampered_path = tampered_root / "pixal_batch_manifest.json"
    tampered_root.chmod(0o755)
    tampered_path.chmod(0o644)
    tampered = contracts.load_json(tampered_path)
    tampered["copied_artifacts"]["pixal_input_rgba"]["sha256"] = "0" * 64
    tampered["batch_sha256"] = adopter._hash_without(tampered, "batch_sha256")
    _write_json(tampered_path, tampered)
    with pytest.raises(contracts.ContractError, match="copied artifact"):
        adopter.load_adopted_batch(tampered_path)


def test_legacy_flux_reference_label_is_preserved():
    assert (
        static_reviews._reference_label_for_batch(
            {"schema": static_reviews.pixal_runner.BATCH_SCHEMA}
        )
        == "approved FLUX.2"
    )


def test_legacy_flux_v1_batch_loader_behavior_is_preserved(tmp_path):
    output = tmp_path / "legacy" / INSTANCE_ID / "pixal_raw_1024.glb"
    _write_glb(output)
    batch = {
        "schema": static_reviews.pixal_runner.BATCH_SCHEMA,
        "status": "passed_generation_and_glb_readback",
        "job_count": 1,
        "attempts": [
            {
                "instance_id": INSTANCE_ID,
                "output": {
                    "path": f"{INSTANCE_ID}/pixal_raw_1024.glb",
                    "sha256": adopter._sha256_file(output),
                    "size_bytes": output.stat().st_size,
                },
            }
        ],
        "automatic_checks": {"overall": "passed"},
    }
    batch["batch_sha256"] = adopter._hash_without(batch, "batch_sha256")
    batch_path = tmp_path / "legacy" / "pixal_batch_manifest.json"
    _write_json(batch_path, batch)

    loaded_path, loaded = static_reviews.load_pixal_batch(batch_path)

    assert loaded_path == batch_path.resolve()
    assert loaded == batch
