#!/usr/bin/env python3
"""Execute agent-passed Route-2 color fastlane references with pinned Pixal3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import route2_controlled_color_references_v3 as color
from tools import route2_controlled_color_fastlane_decision_v1 as decisions
from tools import route2_controlled_geometry_pixal_preflight_v1 as executor


SCHEMA = "route2_controlled_color_pixal_execution_contract_v1"
MANIFEST_SCHEMA = "route2_controlled_color_pixal_candidate_v1"
ATTEMPT_SCHEMA = "route2_controlled_color_pixal_attempt_v1"
RUNNER_PATH = Path(__file__).resolve()
SOURCE_PIXAL_JOBS = color.OUTPUT_ROOT / "fastlane_review_summary_v1/pixal_jobs_v1.json"
SOURCE_PIXAL_JOBS_SHA256 = "1214e88d3aae86a38f015a07aa77ca896d288bf269dd39cb7041d7011ceada97"
OUTPUT_ROOT = color.PIXAL_OUTPUT_ROOT


class ColorPixalError(RuntimeError):
    pass


def source_jobs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = executor._load_json(SOURCE_PIXAL_JOBS, "color fastlane Pixal jobs", mode=0o444)
    if executor.sha256_file(SOURCE_PIXAL_JOBS) != SOURCE_PIXAL_JOBS_SHA256:
        raise ColorPixalError("color fastlane Pixal jobs SHA-256 changed")
    jobs = payload.get("jobs")
    expected_ids = [
        f"route2_color_v3_{case_id}"
        for case_id in color.CASE_BY_ID
        if (color.OUTPUT_ROOT / "cases" / case_id / decisions.DECISION_FILENAME).is_file()
        and json.loads(
            (color.OUTPUT_ROOT / "cases" / case_id / decisions.DECISION_FILENAME).read_text(
                encoding="utf-8"
            )
        ).get("status")
        == "agent_2d_fastlane_passed"
    ]
    if (
        payload.get("schema") != decisions.PIXAL_JOBS_SCHEMA
        or payload.get("state_classification") != "research_candidate_fastlane"
        or payload.get("formal_registration_authorized") is not False
        or not isinstance(jobs, list)
        or [job.get("asset_id") for job in jobs if isinstance(job, Mapping)] != expected_ids
    ):
        raise ColorPixalError("color fastlane Pixal jobs schema/order changed")
    validated = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ColorPixalError("color Pixal job is not an object")
        asset_id = str(job["asset_id"])
        case_id = asset_id[len("route2_color_v3_") :]
        root, candidate = color._load_candidate(case_id)
        decision_path = root / decisions.DECISION_FILENAME
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        expected_output = OUTPUT_ROOT / asset_id
        if (
            decision.get("schema") != decisions.SCHEMA
            or decision.get("status") != "agent_2d_fastlane_passed"
            or decision.get("pixal_authorized") is not True
            or job.get("base_asset_id") != candidate.get("base_asset_id")
            or job.get("attribute") != candidate.get("attribute")
            or job.get("target_color_name") != candidate.get("target_color_name")
            or job.get("state_classification") != "research_candidate_fastlane"
            or job.get("input_rgba") != candidate.get("artifacts", {}).get("candidate_rgba.png")
            or job.get("reference_manifest") != executor._record(root / "candidate_manifest.json")
            or job.get("reference_decision") != executor._record(decision_path)
            or job.get("model")
            != {"name": "TencentARC/Pixal3D", "revision": executor.PIXAL_REVISION}
            or job.get("parameters") != executor.PARAMETERS
            or Path(str(job.get("output_dir"))).absolute() != expected_output
            or job.get("execution_status") != "ready_for_pixal_color_fastlane_preflight"
        ):
            raise ColorPixalError(f"color Pixal job lineage changed: {case_id}")
        executor._require_record(
            Path(job["input_rgba"]["path"]), job["input_rgba"], f"{case_id} color RGBA"
        )
        validated.append({
            **job,
            # The delegated immutable executor records this generic slot.  It
            # carries the exact color attribute; no geometry reinterpretation occurs.
            "geometry_attribute": f"color_only:{job['attribute']}",
        })
    return payload, validated


def configure_executor() -> None:
    executor.SCHEMA = SCHEMA
    executor.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    executor.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    executor.RUNNER_PATH = RUNNER_PATH
    executor.SOURCE_ROOT = color.OUTPUT_ROOT
    executor.SOURCE_PIXAL_JOBS = SOURCE_PIXAL_JOBS
    executor.SOURCE_PIXAL_JOBS_SHA256 = SOURCE_PIXAL_JOBS_SHA256
    executor.OUTPUT_ROOT = OUTPUT_ROOT
    executor.geometry = color
    executor._source_jobs = source_jobs


def prepare() -> Path:
    configure_executor()
    return executor.prepare()


def run(asset_id: str, gpu: str) -> Path:
    configure_executor()
    return executor.run(asset_id, gpu)


def asset_ids() -> tuple[str, ...]:
    return tuple(job["asset_id"] for job in source_jobs()[1])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--asset-id", choices=asset_ids(), required=True)
    run_parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        print(f"ROUTE2_CONTROLLED_COLOR_PIXAL_PREPARED {prepare()}")
    else:
        print(f"ROUTE2_CONTROLLED_COLOR_PIXAL_PUBLISHED {run(args.asset_id, args.gpu)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
