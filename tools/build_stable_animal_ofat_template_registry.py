#!/usr/bin/env python3
"""Publish a generic stable-template registry from OFAT + UE import evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import finalize_stable_animal_ofat_review as review_lib  # noqa: E402


SCHEMA = "avengine_stable_animal_template_registry_v2"
JOBS_SCHEMA = "stable_animal_ue_import_batch_v1"
RESULT_SCHEMA = "stable_animal_ue_import_result_v1"
PENDING_REVIEW_STATUS = "local_ofat_visual_review_pending"


class RegistryError(RuntimeError):
    """Raised when one registry authority cannot be authenticated."""


def artifact(path: Path) -> dict[str, Any]:
    path = review_lib.regular_file(path, "registry artifact")
    return {
        "path": str(path),
        "sha256": review_lib.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def absolute_from_root(root_id: str, relative: str) -> Path:
    roots = {"spear_repo": SPEAR_ROOT, "avengine_repo": AVENGINE_ROOT}
    if root_id not in roots:
        raise RegistryError(f"unsupported artifact root: {root_id}")
    path = (roots[root_id] / relative).resolve()
    try:
        path.relative_to(roots[root_id].resolve())
    except ValueError as error:
        raise RegistryError(f"artifact escaped root: {relative}") from error
    return path


def authenticate(args: argparse.Namespace):
    review_path, review = review_lib.load_json(args.review, "OFAT review")
    if (
        review.get("schema") != review_lib.SCHEMA
        or review.get("manifest_sha256") != contracts.manifest_sha256(review)
        or review.get("formal_dataset_registration_authorized") is not False
    ):
        raise RegistryError("OFAT review hash or classification failed")
    profile_path, raw_profile = review_lib.load_json(
        Path(review["profile"]["absolute_path"]), "attribute profile"
    )
    profile = contracts.validate_attribute_profile(raw_profile)
    if (
        review_lib.sha256_file(profile_path) != review["profile"]["sha256"]
        or contracts.profile_sha256(profile) != review["profile"]["profile_sha256"]
    ):
        raise RegistryError("attribute profile changed after OFAT review")
    jobs_path, jobs = review_lib.load_json(args.jobs, "UE import jobs")
    result_path, result = review_lib.load_json(args.result, "UE import result")
    if (
        jobs.get("schema") != JOBS_SCHEMA
        or jobs.get("job_count") != len(jobs.get("jobs", []))
        or jobs.get("source_contract", {}).get("review_manifest_sha256")
        != review["manifest_sha256"]
        or result.get("schema") != RESULT_SCHEMA
        or Path(result.get("input_manifest", "")).resolve() != jobs_path
        or result.get("input_manifest_sha256") != review_lib.sha256_file(jobs_path)
        or result.get("passed_count") != len(result.get("results", []))
    ):
        raise RegistryError("UE jobs/result are not authenticated to this OFAT review")
    provenance_path, provenance = review_lib.load_json(
        args.base_provenance, "base-template provenance"
    )
    base_artifact = profile["base_template"]["artifact"]
    if (
        provenance.get("output", {}).get("sha256") != base_artifact["sha256"]
        or provenance.get("license", {}).get("spdx_id") != "MIT"
    ):
        raise RegistryError("Rocketbox base-template/license provenance mismatch")
    license_path = review_lib.regular_file(
        Path(provenance["license"]["evidence_path"]), "Rocketbox license"
    )
    if review_lib.sha256_file(license_path) != provenance["license"]["evidence_sha256"]:
        raise RegistryError("Rocketbox license snapshot changed")
    motion_path, motion = review_lib.load_json(
        args.motion_provenance, "motion provenance"
    )
    source_rig = motion.get("source_rig", {})
    source_rig_path = absolute_from_root(source_rig["root_id"], source_rig["path"])
    license_record = motion.get("license_snapshot", {})
    motion_license_path = absolute_from_root(
        license_record["root_id"], license_record["path"]
    )
    for path, record, label in (
        (source_rig_path, source_rig, "motion source"),
        (motion_license_path, license_record, "motion license"),
    ):
        path = review_lib.regular_file(path, label)
        if review_lib.sha256_file(path) != record["sha256"]:
            raise RegistryError(f"{label} snapshot changed")
    return (
        review_path,
        review,
        profile_path,
        profile,
        jobs_path,
        jobs,
        result_path,
        result,
        provenance_path,
        provenance,
        motion_path,
        motion,
    )


def build_entries(
    review: dict[str, Any],
    profile: dict[str, Any],
    jobs: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews = {item["instance_id"]: item for item in review["entries"]}
    jobs_by_id = {item["template_id"]: item for item in jobs["jobs"]}
    results = {item["template_id"]: item for item in result["results"]}
    if set(reviews) != set(jobs_by_id) or set(reviews) != set(results):
        raise RegistryError("OFAT/UE identity coverage differs")
    entries = []
    for instance_id in sorted(reviews):
        source = reviews[instance_id]
        job = jobs_by_id[instance_id]
        imported = results[instance_id]
        glb = source["realization"]["glb"]
        deformation = source["deformation"]["artifact"]
        direction = job["direction_contract"]
        if (
            imported.get("asset_id") != instance_id
            or imported.get("tag") != job["tag"]
            or imported.get("source_sha256") != glb["sha256"]
            or set(imported.get("actions", [])) != {"Walking", "Idle"}
            or imported.get("status") != "passed"
            or imported.get("human_review_status") != PENDING_REVIEW_STATUS
            or direction.get("automatic_fine_yaw_inference") is not False
            or direction.get("ue_visual_review") != "pending"
        ):
            raise RegistryError(f"UE entry contract failed: {instance_id}")
        entries.append(
            {
                "template_id": instance_id,
                "taxonomy_label": job["taxonomy_label"],
                "route_id": "stable_controlled_instance_template_v1",
                "state_classification": "research_candidate",
                "formal_dataset_registration_authorized": False,
                "runtime_glb": {
                    "path": glb["absolute_path"],
                    "sha256": glb["sha256"],
                    "size_bytes": glb["size_bytes"],
                },
                "realization_manifest": source["realization"]["manifest"],
                "deformation_audit": {
                    "path": deformation["absolute_path"],
                    "sha256": deformation["sha256"],
                    "size_bytes": deformation["size_bytes"],
                },
                "media": {
                    action.lower() + "_side": {
                        "path": source["videos"][action]["absolute_path"],
                        "sha256": source["videos"][action]["sha256"],
                        "size_bytes": source["videos"][action]["size_bytes"],
                    }
                    for action in ("Walking", "Idle")
                },
                "actions": ["Walking", "Idle"],
                "direction": {
                    "authored_front_axis": profile["rig_profile"]["front_axis"],
                    "runtime_front_axis": "positive_x",
                    "cardinal_yaw_deg": direction["cardinal_yaw_deg"],
                    "automatic_fine_yaw_inference": False,
                    "review_status": PENDING_REVIEW_STATUS,
                },
                "sampled_attributes": source["sampled_attributes"],
                "fixed_attributes": source["fixed_attributes"],
                "target_physical_profile": source["target_physical_profile"],
                "qa": {
                    "glb_readback": "passed",
                    "topology_uv_skin_preserved": "passed",
                    "walking_deformation": "passed_automatic_deformation_measurements",
                    "idle_deformation": "passed_automatic_deformation_measurements",
                    "isolated_media": "passed_agent_visual_check",
                    "ue_import": "passed",
                    "ue_apartment_media": "pending",
                    "human_visual_review": "pending",
                    "audio": "pending",
                },
                "ue_import": {
                    "tag": imported["tag"],
                    "blueprint": imported["blueprint"],
                    "asset_count": imported["asset_count"],
                    "assets": imported["assets"],
                },
            }
        )
    return entries


def publish(args: argparse.Namespace) -> Path:
    (
        review_path,
        review,
        profile_path,
        profile,
        jobs_path,
        jobs,
        result_path,
        result,
        provenance_path,
        provenance,
        motion_path,
        motion,
    ) = authenticate(args)
    entries = build_entries(review, profile, jobs, result)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate_pending_ue_apartment_audio_human_review",
        "formal_dataset_registration_authorized": False,
        "source_review": artifact(review_path),
        "attribute_profile": artifact(profile_path),
        "ue_import_jobs": artifact(jobs_path),
        "ue_import_result": artifact(result_path),
        "base_template_provenance": artifact(provenance_path),
        "motion_provenance": artifact(motion_path),
        "licenses": {
            "rocketbox_geometry_material_skeleton": {
                "spdx_id": provenance["license"]["spdx_id"],
                "evidence": {
                    "path": provenance["license"]["evidence_path"],
                    "sha256": provenance["license"]["evidence_sha256"],
                },
            },
            "quaternius_motion": {
                "spdx_id": motion["license_snapshot"]["declared_license"],
                "evidence": motion["license_snapshot"],
            },
        },
        "entry_count": len(entries),
        "ofat": review["ofat"],
        "entries": entries,
        "automatic_checks": {
            "review_profile_provenance_and_licenses_authenticated": True,
            "all_ue_imports_passed": True,
            "all_domain_values_preserved": True,
            "formal_registration_authorized": False,
            "overall": "passed_pending_ue_apartment_audio_human_review",
        },
    }
    payload["manifest_sha256"] = contracts.manifest_sha256(payload)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RegistryError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    observed = json.loads(output.read_text(encoding="utf-8"))
    if observed["manifest_sha256"] != contracts.manifest_sha256(observed):
        raise RegistryError("published registry hash readback failed")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path, required=True)
    parser.add_argument("--motion-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = publish(args)
    except (RegistryError, review_lib.ReviewError, OSError, ValueError) as error:
        print(f"STABLE_ANIMAL_OFAT_REGISTRY_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_ANIMAL_OFAT_REGISTRY_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
