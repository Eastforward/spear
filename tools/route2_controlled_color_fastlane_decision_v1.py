#!/usr/bin/env python3
"""Record visual fastlane decisions for immutable FLUX.2 color canaries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import route2_controlled_color_references_v3 as color
from tools import route2_controlled_geometry_references_v3 as geometry


SCHEMA = "route2_controlled_color_fastlane_agent_qa_v1"
SUMMARY_SCHEMA = "route2_controlled_color_fastlane_summary_v1"
PIXAL_JOBS_SCHEMA = "route2_controlled_color_fastlane_pixal_jobs_v1"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = color.OUTPUT_ROOT
DECISION_FILENAME = "agent_2d_fastlane_qa.json"
ADVISORY_CHECKS = {"texture_edges_retained", "texture_luminance_retained"}
CRITICAL_CHECKS = {
    "outside_mask_rgb_exact",
    "source_alpha_byte_identical",
    "non_target_guard_byte_identical",
    "semantic_core_changed",
    "natural_target_color_close",
    "floor_contact_unchanged",
}


class FastlaneDecisionError(RuntimeError):
    pass


def _record(path: Path, *, mode: int | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        raise FastlaneDecisionError(f"artifact must be a direct nonempty file: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise FastlaneDecisionError(f"artifact mode changed: {path}")
    return {"path": str(path), "sha256": geometry.sha256_file(path), "size_bytes": path.stat().st_size}


def _candidate(case_id: str) -> tuple[Path, dict[str, Any]]:
    root = OUTPUT_ROOT / "cases" / case_id
    path = root / "candidate_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        case_id not in color.CASE_BY_ID
        or manifest.get("schema") != color.CANDIDATE_SCHEMA
        or manifest.get("case_id") != case_id
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("runner") != _record(color.RUNNER_PATH)
    ):
        raise FastlaneDecisionError("color candidate identity/state/runner changed")
    for filename, value in manifest.get("artifacts", {}).items():
        if value != _record(root / filename, mode=0o444):
            raise FastlaneDecisionError(f"color candidate artifact changed: {filename}")
    checks = manifest.get("metrics", {}).get("checks")
    if not isinstance(checks, Mapping) or not CRITICAL_CHECKS <= set(checks):
        raise FastlaneDecisionError("color candidate checks are incomplete")
    return root, manifest


def advisory_only_automatic_rejection(manifest: Mapping[str, Any]) -> bool:
    checks = manifest.get("metrics", {}).get("checks", {})
    failed = {name for name, passed in checks.items() if passed is not True}
    return (
        manifest.get("automatic_2d_gate") == "rejected"
        and all(checks.get(name) is True for name in CRITICAL_CHECKS)
        and failed
        and failed <= ADVISORY_CHECKS
    )


def review(case_id: str, status: str, notes: str) -> Path:
    if status not in {"agent_2d_fastlane_passed", "rejected"} or not notes.strip():
        raise FastlaneDecisionError("decision or notes are invalid")
    root, manifest = _candidate(case_id)
    passed = status == "agent_2d_fastlane_passed"
    if passed and not advisory_only_automatic_rejection(manifest):
        raise FastlaneDecisionError("fastlane can override only texture-correlation advisories")
    destination = root / DECISION_FILENAME
    payload = {
        "schema": SCHEMA,
        "case_id": case_id,
        "status": status,
        "state_classification": "research_candidate_fastlane" if passed else "rejected",
        "candidate_manifest": _record(root / "candidate_manifest.json", mode=0o444),
        "contact_sheet": manifest["artifacts"]["contact_sheet.png"],
        "automatic_2d_gate": manifest["automatic_2d_gate"],
        "automatic_check_disposition": {
            name: ("hard_pass" if value is True else "preserved_texture_correlation_advisory")
            for name, value in manifest["metrics"]["checks"].items()
        },
        "visual_checks": {
            "only_target_semantic_region_changed": passed,
            "target_color_is_clear_and_natural": passed,
            "identity_face_pose_camera_preserved": passed,
            "target_geometry_and_silhouette_preserved": passed,
            "candidate_remains_pixal_bindable": passed,
        },
        "notes": notes.strip(),
        "reviewer": "codex_female_route2_base",
        "reviewer_kind": "agent",
        "pixal_authorized": passed,
        "user_acceptance": "not_claimed",
        "formal_dataset_registration_authorized": False,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        os.write(descriptor, geometry._json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def finalize() -> Path:
    destination = OUTPUT_ROOT / "fastlane_review_summary_v1"
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    records, panels, jobs = [], [], []
    for case_id in color.CASE_BY_ID:
        root, manifest = _candidate(case_id)
        decision_path = root / DECISION_FILENAME
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            decision.get("schema") != SCHEMA
            or decision.get("case_id") != case_id
            or decision.get("candidate_manifest") != _record(root / "candidate_manifest.json", mode=0o444)
        ):
            raise FastlaneDecisionError(f"fastlane decision changed: {case_id}")
        with Image.open(root / "contact_sheet.png") as opened:
            panels.append((case_id, opened.convert("RGB")))
        records.append({
            "case_id": case_id,
            "decision": _record(decision_path, mode=0o444),
            "status": decision["status"],
        })
        if decision["status"] == "agent_2d_fastlane_passed":
            jobs.append({
                "asset_id": f"route2_color_v3_{case_id}",
                "base_asset_id": manifest["base_asset_id"],
                "attribute": manifest["attribute"],
                "target_color_name": manifest["target_color_name"],
                "state_classification": "research_candidate_fastlane",
                "input_rgba": manifest["artifacts"]["candidate_rgba.png"],
                "reference_manifest": _record(root / "candidate_manifest.json", mode=0o444),
                "reference_decision": _record(decision_path, mode=0o444),
                "model": {"name": "TencentARC/Pixal3D", "revision": geometry.PIXAL_REVISION},
                "parameters": {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True},
                "output_dir": str(color.PIXAL_OUTPUT_ROOT / f"route2_color_v3_{case_id}"),
                "execution_status": "ready_for_pixal_color_fastlane_preflight",
            })
    staging = Path(tempfile.mkdtemp(prefix=".fastlane_review_summary_v1.", suffix=".staging", dir=OUTPUT_ROOT))
    try:
        geometry._write_image(staging / "all_cases_contact_sheet.png", geometry.make_contact_sheet(panels))
        (staging / "pixal_jobs_v1.json").write_bytes(geometry._json_bytes({
            "schema": PIXAL_JOBS_SCHEMA,
            "state_classification": "research_candidate_fastlane",
            "formal_registration_authorized": False,
            "source_jobs_contract": _record(OUTPUT_ROOT / "color_jobs_v3.json", mode=0o444),
            "jobs": jobs,
        }))
        (staging / "summary.json").write_bytes(geometry._json_bytes({
            "schema": SUMMARY_SCHEMA,
            "state_classification": "research_candidate_fastlane",
            "formal_registration_authorized": False,
            "case_count": len(records),
            "passed_count": sum(item["status"] == "agent_2d_fastlane_passed" for item in records),
            "rejected_count": sum(item["status"] == "rejected" for item in records),
            "cases": records,
            "pixal_job_count": len(jobs),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--case-id", choices=tuple(color.CASE_BY_ID), required=True)
    review_parser.add_argument("--status", choices=("agent_2d_fastlane_passed", "rejected"), required=True)
    review_parser.add_argument("--notes", required=True)
    commands.add_parser("finalize")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "review":
        print(f"CONTROLLED_COLOR_FASTLANE_REVIEW_OK {review(args.case_id, args.status, args.notes)}")
    else:
        print(f"CONTROLLED_COLOR_FASTLANE_FINALIZED {finalize()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
