#!/usr/bin/env python3
"""Segment approved FLUX.2 animals and compile authenticated Pixal3D jobs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_isnet_worker as isnet
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import review_controlled_animal_flux2_candidates as review


PIXAL_INPUT_SCHEMA = "avengine_controlled_animal_pixal_inputs_v1"
ISNET_PYTHON = Path("/data/jzy/miniconda3/envs/hunyuan3d/bin/python")
ISNET_WORKER = Path(__file__).resolve().parent / "controlled_animal_isnet_worker.py"
PIXAL_MODEL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    return review._sha256_file(path)


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_review_batch(path: Path):
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"2D review batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != review.BATCH_REVIEW_SCHEMA
        or payload.get("review_batch_sha256")
        != _hash_without(payload, "review_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("2D review batch contract/hash is invalid")
    return payload


def prepare_pixal_inputs(
    review_batch_path: Path,
    output_root: Path,
    pixal_output_root: Path,
) -> Path:
    review_batch = load_review_batch(review_batch_path)
    flux_batch_path = Path(review_batch["flux2_batch"]["path"])
    flux_root, flux_batch, candidates = review.load_flux_batch(flux_batch_path)
    if flux_batch["batch_sha256"] != review_batch["flux2_batch"]["batch_sha256"]:
        raise contracts.ContractError("review and FLUX.2 batch hashes differ")
    approved_reviews = {}
    review_root = Path(review_batch_path).resolve().parent
    for item in review_batch["reviews"]:
        record = item["review"]
        path = (review_root / record["path"]).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise contracts.ContractError("2D review artifact changed")
        payload = contracts.load_json(path)
        if (
            payload.get("schema") != review.REVIEW_SCHEMA
            or payload.get("instance_id") != item["instance_id"]
            or payload.get("candidate", {}).get("sha256") != item["candidate_sha256"]
            or payload.get("review_sha256") != _hash_without(payload, "review_sha256")
        ):
            raise contracts.ContractError("2D review record contract/hash is invalid")
        if payload["decision"] == "approved_for_pixal3d":
            approved_reviews[payload["instance_id"]] = payload
    if len(approved_reviews) != review_batch["approved_count"]:
        raise contracts.ContractError("approved 2D review count changed")
    if not approved_reviews:
        raise contracts.ContractError("no approved candidates for Pixal3D")

    preflight_path = Path(flux_batch["execution_preflight"]["path"])
    preflight = material_execution._load_preflight(preflight_path)
    animal_jobs = {
        job["consumer_requests"][0]["instance_id"]: job
        for job in preflight["routes"]["flux2_pixal3d_animal_v1"]
    }
    output_root = Path(output_root).absolute()
    pixal_output_root = Path(pixal_output_root).absolute()
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
        segmentation_jobs = []
        for instance_id, reviewed in sorted(approved_reviews.items()):
            candidate = candidates[instance_id]
            if reviewed["candidate"]["sha256"] != candidate["index"]["candidate"]["sha256"]:
                raise contracts.ContractError("approved candidate hash changed")
            destination = staging / "segmentation" / instance_id
            segmentation_jobs.append(
                {
                    "instance_id": instance_id,
                    "candidate_path": str(candidate["files"]["candidate"]),
                    "candidate_sha256": reviewed["candidate"]["sha256"],
                    "alpha_path": str(destination / "alpha_isnet.png"),
                    "rgba_path": str(destination / "input_rgba_isnet.png"),
                }
            )
        jobs_payload = {"schema": isnet.JOBS_SCHEMA, "jobs": segmentation_jobs}
        jobs_path = staging / "isnet_jobs.json"
        contracts.write_json_no_replace(jobs_path, jobs_payload)
        status_path = staging / "isnet_status.json"
        log_path = staging / "isnet.log"
        command = [
            str(ISNET_PYTHON),
            str(ISNET_WORKER),
            "--jobs",
            str(jobs_path),
            "--status",
            str(status_path),
        ]
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        if completed.returncode != 0:
            raise contracts.ContractError("ISNet worker failed")
        status = contracts.load_json(status_path)
        if (
            status.get("schema") != isnet.STATUS_SCHEMA
            or status.get("status") != "passed"
            or status.get("passed_count") != len(segmentation_jobs)
            or status.get("failed_count") != 0
        ):
            raise contracts.ContractError("ISNet worker status is incomplete")
        status_by_id = {item["instance_id"]: item for item in status["jobs"]}
        pixal_jobs = []
        segmentations = []
        for job in segmentation_jobs:
            instance_id = job["instance_id"]
            status_item = status_by_id[instance_id]
            alpha_path = Path(job["alpha_path"])
            rgba_path = Path(job["rgba_path"])
            if (
                _sha256_file(alpha_path) != status_item["alpha_sha256"]
                or _sha256_file(rgba_path) != status_item["rgba_sha256"]
            ):
                raise contracts.ContractError("ISNet output hash changed")
            with Image.open(rgba_path) as opened:
                opened.load()
                if opened.mode != "RGBA" or opened.size != (1024, 1024):
                    raise contracts.ContractError("Pixal input RGBA contract changed")
            controlled_job = animal_jobs[instance_id]
            generation = controlled_job["generation_plan"]
            if (
                generation["model_revisions"]["pixal3d"] != PIXAL_MODEL_REVISION
                or generation["model_revisions"]["dino"] != DINO_REVISION
            ):
                raise contracts.ContractError("Pixal/DINO model revision changed")
            public_rgba = (
                output_root / "segmentation" / instance_id / "input_rgba_isnet.png"
            )
            pixal_output = pixal_output_root / instance_id / "pixal_raw_1024.glb"
            candidate_record = candidates[instance_id]["index"]["candidate"]
            pixal_jobs.append(
                {
                    "legacy_tag": instance_id,
                    "candidate_tag": f"{instance_id}_pixal_v1",
                    "rig_mode": "animated_transfer",
                    "seed": int(generation["generation_seed"]),
                    "reference": {
                        "source": {
                            "path": str(candidates[instance_id]["files"]["candidate"]),
                            "sha256": candidate_record["sha256"],
                            "size_bytes": candidate_record["size_bytes"],
                        },
                        "pixal_input": {
                            "path": str(public_rgba),
                            "sha256": _sha256_file(rgba_path),
                            "size_bytes": rgba_path.stat().st_size,
                        },
                        "normalization": "pinned_isnet_general_use_alpha_v1",
                    },
                    "output": str(pixal_output),
                    "manifest": str(pixal_output.with_suffix(".manifest.json")),
                    "controlled_request": {
                        "execution_job_id": controlled_job["execution_job_id"],
                        "instance_id": instance_id,
                        "request_sha256": controlled_job["consumer_requests"][0][
                            "request_sha256"
                        ],
                        "profile_schema_id": controlled_job["profile_schema_id"],
                        "sampled_attributes": controlled_job["sampled_attributes"],
                        "target_physical_profile": controlled_job[
                            "target_physical_profile"
                        ],
                    },
                    "model_revisions": {
                        "pixal3d": PIXAL_MODEL_REVISION,
                        "dino": DINO_REVISION,
                    },
                    "parameters": {
                        "resolution": 1024,
                        "manual_fov": 0.2,
                        "low_vram": True,
                    },
                }
            )
            segmentations.append(
                {
                    "instance_id": instance_id,
                    "candidate_sha256": candidates[instance_id]["index"]["candidate"][
                        "sha256"
                    ],
                    "alpha": _relative_record(alpha_path, staging),
                    "rgba": _relative_record(rgba_path, staging),
                    "foreground_fraction_at_128": status_item[
                        "foreground_fraction_at_128"
                    ],
                    "foreground_bbox_xyxy": status_item["foreground_bbox_xyxy"],
                    "status": "passed",
                }
            )
        pixal_payload: dict[str, Any] = {
            "schema": PIXAL_INPUT_SCHEMA,
            "status": "ready_for_pixal3d",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "review_batch": {
                "path": str(Path(review_batch_path).resolve()),
                "sha256": _sha256_file(Path(review_batch_path)),
                "review_batch_sha256": review_batch["review_batch_sha256"],
            },
            "isnet": {
                "model_path": str(isnet.MODEL_PATH),
                "model_sha256": isnet.MODEL_SHA256,
                "worker": str(ISNET_WORKER),
                "python": str(ISNET_PYTHON),
                "status": _relative_record(status_path, staging),
                "log": _relative_record(log_path, staging),
            },
            "pixal_output_root": str(pixal_output_root),
            "job_count": len(pixal_jobs),
            "jobs": pixal_jobs,
            "segmentations": segmentations,
            "automatic_checks": {
                "all_2d_reviews_approved": True,
                "all_candidate_hashes_reauthenticated": True,
                "all_isnet_segmentations_passed": True,
                "all_pixal_inputs_rgba_1024": True,
                "pixal_and_dino_revisions_pinned": True,
                "overall": "passed",
            },
        }
        pixal_payload["manifest_sha256"] = _json_sha256(pixal_payload)
        contracts.write_json_no_replace(staging / "pixal_inputs_manifest.json", pixal_payload)
        material_execution.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "pixal_inputs_manifest.json"
    except Exception:
        material_execution.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pixal-output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = prepare_pixal_inputs(
            args.review_batch, args.output_root, args.pixal_output_root
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_PIXAL_INPUT_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_PIXAL_INPUT_OK "
        f"jobs={manifest['job_count']} output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
