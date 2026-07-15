#!/usr/bin/env python3
"""Realize and audit nine OFAT instances for every stable quadruped profile."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import prepare_controlled_source_asset_execution as preflight_lib  # noqa: E402


SCHEMA = "avengine_stable_quadruped_ofat_batch_v1"
BLENDER = Path("/data/jzy/.local/bin/blender")
BUILDER = SPEAR_ROOT / "tools/blender_build_stable_quadruped_instance.py"
TEXTURED_BUILDER = SPEAR_ROOT / "tools/blender_build_stable_animal_instance.py"
INVENTORY = SPEAR_ROOT / "tools/blender_inventory_animal_template.py"
DEFORMATION = SPEAR_ROOT / "tools/blender_audit_skinned_deformation.py"
BASELINE = {"size": "medium", "body_build": "standard", "life_stage": "adult"}
ATTRIBUTES = ("size", "body_build", "coat_tone", "life_stage")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def operations(job: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        item["attribute"]: item
        for item in job["stable_instance_plan"]["attribute_operations"]
    }


def builder_for_surface_mode(surface_mode: str) -> Path:
    if surface_mode == "solid_material_pbr":
        return BUILDER
    if surface_mode == "textured_pbr":
        return TEXTURED_BUILDER
    raise contracts.ContractError(f"unsupported stable-animal surface mode: {surface_mode}")


def job_surface_mode(job: Mapping[str, Any]) -> str:
    parameters = operations(job)["coat_tone"]["parameters"]
    # The first textured Beagle profile predates the explicit surface flag.
    return str(parameters.get("surface_mode", "textured_pbr"))


def deformation_gate(deformation: Mapping[str, Any], *, policy: str):
    if policy not in {"strict", "record_only"}:
        raise contracts.ContractError(f"unsupported deformation policy: {policy}")
    strict_passed = deformation.get("overall") == "passed"
    return (
        {
            "deformation_audit_completed": True,
            "deformation_policy_satisfied": strict_passed or policy == "record_only",
        },
        {
            "policy": policy,
            "strict_deformation_passed": strict_passed,
            "strict_overall": deformation.get("overall"),
            "strict_failure_retained": not strict_passed,
        },
    )


def coat_baseline(jobs: Sequence[Mapping[str, Any]]) -> str:
    candidates = set()
    for job in jobs:
        operation = operations(job)["coat_tone"]
        if float(operation["parameters"]["coat_luminance_gain"]) == 1.0:
            candidates.add(operation["value"])
    if len(candidates) != 1:
        raise contracts.ContractError(
            f"expected one coat baseline, got {sorted(candidates)}"
        )
    return candidates.pop()


def instance_id(job: Mapping[str, Any]) -> str:
    consumers = job.get("consumer_requests", [])
    if len(consumers) != 1:
        raise contracts.ContractError("stable Cartesian job must have one consumer")
    return str(consumers[0]["instance_id"])


def select_ofat_jobs(profile_jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(profile_jobs) != 81:
        raise contracts.ContractError(
            f"full four-attribute Cartesian profile must have 81 jobs, got {len(profile_jobs)}"
        )
    baseline = {**BASELINE, "coat_tone": coat_baseline(profile_jobs)}
    matches = {
        tuple(sorted(job["sampled_attributes"].items())): job for job in profile_jobs
    }
    if len(matches) != 81:
        raise contracts.ContractError("profile jobs are not 81 unique absolute combinations")

    def exact(attributes):
        key = tuple(sorted(attributes.items()))
        if key not in matches:
            raise contracts.ContractError(f"missing OFAT combination: {attributes}")
        return matches[key]

    selected = [
        {
            "label": "baseline",
            "changed_attribute_from_baseline": None,
            "job": exact(baseline),
        }
    ]
    all_values = {
        attribute: sorted(
            {job["sampled_attributes"][attribute] for job in profile_jobs}
        )
        for attribute in ATTRIBUTES
    }
    ordered_alternatives = {
        "size": ["small", "large"],
        "body_build": ["slim", "stocky"],
        "life_stage": ["young", "senior"],
    }
    coat_values = []
    for value in all_values["coat_tone"]:
        candidate = next(
            job
            for job in profile_jobs
            if job["sampled_attributes"]["coat_tone"] == value
        )
        gain = float(operations(candidate)["coat_tone"]["parameters"]["coat_luminance_gain"])
        if value != baseline["coat_tone"]:
            coat_values.append((gain, value))
    coat_values.sort(reverse=True)
    ordered_alternatives["coat_tone"] = [value for _, value in coat_values]
    label_prefix = {
        "size": "size",
        "body_build": "build",
        "coat_tone": "coat",
        "life_stage": "age",
    }
    for attribute in ATTRIBUTES:
        alternatives = ordered_alternatives[attribute]
        if len(alternatives) != 2 or set(alternatives) != (
            set(all_values[attribute]) - {baseline[attribute]}
        ):
            raise contracts.ContractError(
                f"invalid OFAT alternatives for {attribute}: {alternatives}"
            )
        for value in alternatives:
            attributes = dict(baseline)
            attributes[attribute] = value
            selected.append(
                {
                    "label": f"{label_prefix[attribute]}_{value}",
                    "changed_attribute_from_baseline": attribute,
                    "job": exact(attributes),
                }
            )
    if len(selected) != 9 or len({instance_id(item["job"]) for item in selected}) != 9:
        raise contracts.ContractError("OFAT selection must produce nine unique instances")
    return selected


def build_plan(preflight: Mapping[str, Any], profile_filters: set[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for job in preflight["routes"]["stable_animal_template_v1"]:
        groups.setdefault(job["profile_schema_id"], []).append(job)
    if profile_filters:
        missing = profile_filters - set(groups)
        if missing:
            raise contracts.ContractError(f"unknown profile filters: {sorted(missing)}")
        groups = {key: value for key, value in groups.items() if key in profile_filters}
    plan = []
    for profile_id in sorted(groups):
        for item in select_ofat_jobs(groups[profile_id]):
            job = item.pop("job")
            plan.append(
                {
                    **item,
                    "profile_schema_id": profile_id,
                    "instance_id": instance_id(job),
                    "taxonomy": job["taxonomy"],
                    "sampled_attributes": job["sampled_attributes"],
                    "acoustic_profile": job["acoustic_profile"],
                    "source_template": job["stable_instance_plan"]["base_template"],
                    "surface_mode": job_surface_mode(job),
                }
            )
    return plan


def run_command(command: Sequence[str], log_path: Path) -> float:
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(list(command), ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(
            list(command),
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}; inspect {log_path}"
        )
    return elapsed


def realize_one(
    entry,
    *,
    preflight_path: Path,
    output_root: Path,
    blender: Path,
    deformation_policy: str,
    license_id: str,
):
    root = output_root / entry["instance_id"]
    root.mkdir(parents=True, exist_ok=False)
    log = root / "pipeline.log"
    glb = root / "instance.glb"
    manifest = root / "manifest.json"
    inventory = root / "inventory.json"
    deformation = root / "deformation.json"
    stage_seconds = {}
    try:
        builder = builder_for_surface_mode(entry["surface_mode"])
        stage_seconds["realize"] = run_command(
            [
                str(blender),
                "--background",
                "--python",
                str(builder),
                "--",
                "--preflight",
                str(preflight_path),
                "--instance-id",
                entry["instance_id"],
                "--output-glb",
                str(glb),
                "--manifest",
                str(manifest),
            ],
            log,
        )
        stage_seconds["inventory"] = run_command(
            [
                str(blender),
                "--background",
                "--python",
                str(INVENTORY),
                "--",
                "--input",
                str(glb),
                "--output",
                str(inventory),
                "--license-id",
                license_id,
            ],
            log,
        )
        stage_seconds["deformation"] = run_command(
            [
                str(blender),
                "--background",
                "--python",
                str(DEFORMATION),
                "--",
                "--input",
                str(glb),
                "--output",
                str(deformation),
                "--action",
                "Walking",
                "--action",
                "Idle",
                "--samples",
                "24",
            ],
            log,
        )
        realized = json.loads(manifest.read_text(encoding="utf-8"))
        inspected = json.loads(inventory.read_text(encoding="utf-8"))
        deformed = json.loads(deformation.read_text(encoding="utf-8"))
        deformation_checks, deformation_record = deformation_gate(
            deformed, policy=deformation_policy
        )
        checks = {
            "topology_uv_skin_unchanged": realized["realization"]["topology_uv_skin_unchanged"],
            "actions_unchanged": realized["realization"]["actions_unchanged"],
            "runtime_front_axis_positive_x": (
                realized["realization"]["runtime_front_axis"] == "positive_x"
            ),
            "no_automatic_fine_yaw": (
                realized["realization"]["automatic_fine_yaw_inference"] is False
            ),
            "glb_has_walk": inspected["has_walk_action"],
            "glb_has_idle": inspected["has_idle_action"],
            **deformation_checks,
        }
        if not all(checks.values()):
            raise RuntimeError(f"post-export checks failed: {checks}")
        return {
            **entry,
            "status": "passed",
            "stage_seconds": stage_seconds,
            "checks": checks,
            "deformation_gate": deformation_record,
            "artifacts": {
                "glb": {"path": str(glb), "sha256": sha256_file(glb)},
                "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
                "inventory": {"path": str(inventory), "sha256": sha256_file(inventory)},
                "deformation": {
                    "path": str(deformation),
                    "sha256": sha256_file(deformation),
                },
            },
        }
    except Exception as error:
        return {
            **entry,
            "status": "failed",
            "stage_seconds": stage_seconds,
            "error": str(error),
            "log": str(log),
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preflight", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--blender", type=Path, default=BLENDER)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--profile-id", action="append", default=[])
    result.add_argument(
        "--deformation-policy",
        choices=("strict", "record_only"),
        default="strict",
        help=(
            "strict rejects metric failures; record_only retains the strict audit "
            "but permits a separate lenient visual gate"
        ),
    )
    result.add_argument(
        "--license-id",
        default="research_candidate_provenance_review_required",
        help="License/provenance label written into inventory evidence.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    preflight_path = args.preflight.resolve()
    output_root = args.output_root.resolve()
    if args.workers < 1 or args.workers > 16:
        print("STABLE_QUADRUPED_OFAT_FAILED workers must be in [1, 16]", file=sys.stderr)
        return 2
    if not args.blender.is_file() or not all(
        path.is_file()
        for path in (BUILDER, TEXTURED_BUILDER, INVENTORY, DEFORMATION)
    ):
        print("STABLE_QUADRUPED_OFAT_FAILED required executable/script missing", file=sys.stderr)
        return 2
    if output_root.exists() or output_root.is_symlink():
        print(f"STABLE_QUADRUPED_OFAT_FAILED refusing to replace {output_root}", file=sys.stderr)
        return 2
    try:
        preflight = preflight_lib.validate_execution_preflight(
            json.loads(preflight_path.read_text(encoding="utf-8"))
        )
        plan = build_plan(preflight, set(args.profile_id))
        output_root.mkdir(parents=True)
        plan_payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "preflight": {
                "path": str(preflight_path),
                "sha256": sha256_file(preflight_path),
                "preflight_sha256": preflight["preflight_sha256"],
            },
            "workers": args.workers,
            "deformation_policy": args.deformation_policy,
            "license_id": args.license_id,
            "profile_count": len({item["profile_schema_id"] for item in plan}),
            "instance_count": len(plan),
            "entries": plan,
        }
        plan_payload["manifest_sha256"] = contracts.manifest_sha256(plan_payload)
        write_json_no_replace(output_root / "batch_plan.json", plan_payload)
    except Exception as error:
        print(f"STABLE_QUADRUPED_OFAT_FAILED {error}", file=sys.stderr)
        return 2

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                realize_one,
                entry,
                preflight_path=preflight_path,
                output_root=output_root,
                blender=args.blender.resolve(),
                deformation_policy=args.deformation_policy,
                license_id=args.license_id,
            ): entry
            for entry in plan
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            print(
                "STABLE_QUADRUPED_OFAT_PROGRESS "
                f"completed={completed}/{len(plan)} status={result['status']} "
                f"instance={result['instance_id']}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    results.sort(key=lambda item: (item["profile_schema_id"], item["label"]))
    passed = sum(item["status"] == "passed" for item in results)
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "batch_plan": {
            "path": str(output_root / "batch_plan.json"),
            "sha256": sha256_file(output_root / "batch_plan.json"),
        },
        "elapsed_seconds": elapsed,
        "deformation_policy": args.deformation_policy,
        "license_id": args.license_id,
        "instance_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "entries": results,
    }
    status["manifest_sha256"] = contracts.manifest_sha256(status)
    write_json_no_replace(output_root / "batch_status.json", status)
    print(
        "STABLE_QUADRUPED_OFAT_DONE "
        f"passed={passed} failed={len(results)-passed} elapsed={elapsed:.2f}s "
        f"output={output_root}",
        flush=True,
    )
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
