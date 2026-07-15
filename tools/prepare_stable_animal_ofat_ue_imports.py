#!/usr/bin/env python3
"""Prepare non-overwriting UE import jobs from an authenticated animal OFAT review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts
from tools import finalize_stable_animal_ofat_review as review_lib


BATCH_SCHEMA = "stable_animal_ue_import_batch_v1"
PREPARATION_SCHEMA = "stable_animal_ofat_ue_import_preparation_v1"
CARDINAL_YAWS = {-180.0, -90.0, 0.0, 90.0, 180.0}


class PreparationError(RuntimeError):
    """Raised when an OFAT result is not ready for the UE research gate."""


def safe_token(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise PreparationError(f"empty safe token from {value!r}")
    return result


def artifact(path: Path) -> dict[str, Any]:
    path = review_lib.regular_file(path, "UE import authority")
    return {
        "path": str(path),
        "sha256": review_lib.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def authenticate_review(path: Path) -> tuple[Path, dict[str, Any]]:
    path, review = review_lib.load_json(path, "OFAT review")
    if (
        review.get("schema") != review_lib.SCHEMA
        or review.get("manifest_sha256") != contracts.manifest_sha256(review)
        or review.get("formal_dataset_registration_authorized") is not False
        or review.get("automatic_checks", {}).get("overall")
        != "passed_pending_human_ue_audio_review"
        or review.get("entry_count") != len(review.get("entries", []))
    ):
        raise PreparationError("OFAT review is not authenticated for UE import")
    return path, review


def build_jobs(
    review: dict[str, Any],
    *,
    species: str,
    breed: str,
    actor_scale: float,
    audio_lookup: str,
    audio_height: float,
    walking_yaw: float,
) -> list[dict[str, Any]]:
    if (
        not math.isfinite(actor_scale)
        or not 0.01 <= actor_scale <= 2.0
        or not math.isfinite(audio_height)
        or not 0.05 <= audio_height <= 3.0
        or walking_yaw not in CARDINAL_YAWS
    ):
        raise PreparationError("invalid scale, audio height, or non-cardinal direction")
    species = safe_token(species)
    breed = safe_token(breed)
    audio_lookup = safe_token(audio_lookup)
    jobs = []
    for entry in review["entries"]:
        qa = entry.get("qa", {})
        if (
            qa.get("glb_readback") != "passed"
            or qa.get("topology_uv_skin_preserved") != "passed"
            or qa.get("actions_preserved") != "passed"
            or qa.get("walking_deformation") != "passed"
            or qa.get("idle_deformation") != "passed"
            or qa.get("ue_apartment") != "pending"
            or qa.get("audio") != "pending"
        ):
            raise PreparationError(f"OFAT entry is not UE-ready: {entry.get('instance_id')}")
        instance_id = safe_token(str(entry["instance_id"]))
        glb = entry["realization"]["glb"]
        glb_path = review_lib.regular_file(Path(glb["absolute_path"]), "instance GLB")
        if (
            glb.get("sha256") != review_lib.sha256_file(glb_path)
            or glb.get("size_bytes") != glb_path.stat().st_size
        ):
            raise PreparationError(f"instance GLB hash mismatch: {instance_id}")
        deformation = entry["deformation"]["artifact"]
        deformation_path = review_lib.regular_file(
            Path(deformation["absolute_path"]), "deformation audit"
        )
        if deformation.get("sha256") != review_lib.sha256_file(deformation_path):
            raise PreparationError(f"deformation hash mismatch: {instance_id}")
        tag = f"stable_{species}_{breed}_{instance_id}"
        jobs.append(
            {
                "asset_id": entry["instance_id"],
                "template_id": entry["instance_id"],
                "tag": tag,
                "taxonomy_label": f"{breed} {entry['label']}",
                "species": species,
                "breed": breed,
                "sampled_attributes": entry["sampled_attributes"],
                "fixed_attributes": entry["fixed_attributes"],
                "target_physical_profile": entry["target_physical_profile"],
                "actor_scale": actor_scale,
                "audio_lookup": audio_lookup,
                "audio_source_height_offset_m": audio_height,
                "walking_forward_yaw_offset_deg": walking_yaw,
                "direction_contract": {
                    "cardinal_yaw_deg": walking_yaw,
                    "automatic_fine_yaw_inference": False,
                    "authority": "explicit_cli_cardinal_value",
                    "ue_visual_review": "pending",
                },
                "rigged_glb": str(glb_path),
                "rigged_glb_sha256": glb["sha256"],
                "expected_actions": ["Idle", "Walking"],
                "deformation_audit": {
                    "path": str(deformation_path),
                    "sha256": deformation["sha256"],
                    "size_bytes": deformation_path.stat().st_size,
                },
                "human_review_status": "local_ofat_visual_review_pending",
                "formal_dataset_registration_authorized": False,
            }
        )
    if len(jobs) != review["entry_count"] or len({item["tag"] for item in jobs}) != len(jobs):
        raise PreparationError("UE job coverage or tag uniqueness failed")
    return jobs


def prepare(args: argparse.Namespace) -> Path:
    review_path, review = authenticate_review(args.review)
    jobs = build_jobs(
        review,
        species=args.species,
        breed=args.breed,
        actor_scale=args.actor_scale,
        audio_lookup=args.audio_lookup,
        audio_height=args.audio_source_height_offset_m,
        walking_yaw=args.walking_forward_yaw_offset_deg,
    )
    output_root = args.output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise PreparationError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        review_record = artifact(review_path)
        batch = {
            "schema": BATCH_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "non_destructive_policy": (
                "unique stable_* content directories per instance; preserve the "
                "base template, all earlier realizations, and all UE revisions"
            ),
            "source_contract": {
                "schema": "authenticated_stable_animal_ofat_review_v1",
                "review": review_record,
                "review_manifest_sha256": review["manifest_sha256"],
                "all_domain_values_realized": review["ofat"][
                    "all_domain_values_realized"
                ],
            },
            # Retain these conventional artifact fields for existing UE tooling.
            "registry": review_record,
            "selection": review_record,
            "job_count": len(jobs),
            "jobs": jobs,
        }
        jobs_path = staging / "ue_import_jobs.json"
        jobs_path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published_jobs = output_root / jobs_path.name
        preparation = {
            "schema": PREPARATION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "jobs": {
                **artifact(jobs_path),
                "path": str(published_jobs),
            },
            "job_count": len(jobs),
            "automatic_checks": {
                "ofat_review_reauthenticated": True,
                "all_glbs_reauthenticated": True,
                "all_walk_idle_deformation_passed": True,
                "fine_yaw_inference_disabled": True,
                "all_domain_values_covered": True,
                "formal_registration_authorized": False,
            },
        }
        (staging / "ue_import_preparation_manifest.json").write_text(
            json.dumps(preparation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return output_root / "ue_import_jobs.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--breed", required=True)
    parser.add_argument("--actor-scale", type=float, required=True)
    parser.add_argument("--audio-lookup", required=True)
    parser.add_argument("--audio-source-height-offset-m", type=float, required=True)
    parser.add_argument("--walking-forward-yaw-offset-deg", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = prepare(args)
    except (PreparationError, review_lib.ReviewError, OSError, ValueError) as error:
        print(f"STABLE_ANIMAL_OFAT_UE_PREP_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_ANIMAL_OFAT_UE_PREP_OK jobs={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
