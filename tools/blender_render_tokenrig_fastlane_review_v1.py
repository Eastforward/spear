#!/usr/bin/env python3
"""Render a research-fastlane Walk/Idle bundle through the strict media core.

Only the in-memory action-status spelling is projected to the historical
renderer contract.  The source manifest must explicitly preserve its strict
edge-stretch rejection, non-formal classification, and numeric advisories.
All GLB, PBR, skeleton, action, FPS, media, and upstream hash checks remain in
the unchanged renderer.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_render_tokenrig_human_review as renderer
from tools import blender_retarget_fastlane_v1 as retarget_fastlane


TRACK = retarget_fastlane.TRACK
EDGE_POLICY = retarget_fastlane.EDGE_POLICY


class FastlaneReviewError(renderer.ReviewRenderError):
    """The retarget bundle is not an authenticated research fastlane."""


def project_fastlane_manifest_for_strict_renderer(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != renderer.RETARGET_SCHEMA:
        raise FastlaneReviewError("fastlane retarget schema is invalid")
    if payload.get("execution_track") != TRACK:
        raise FastlaneReviewError("retarget is not the research fastlane track")
    if payload.get("formal_dataset_asset") is not False:
        raise FastlaneReviewError("fastlane must not claim a formal dataset asset")
    if (
        payload.get("automatic_checks") != "passed"
        or payload.get("automatic_check_scope") != "fastlane_hard_gates_only"
        or payload.get("strict_formal_registration_status")
        != "blocked_by_recorded_edge_stretch"
    ):
        raise FastlaneReviewError("fastlane hard-gate/strict-registration state is invalid")
    strict_failure = payload.get("strict_failure_evidence")
    if not isinstance(strict_failure, Mapping) or any(
        not strict_failure.get(field) for field in ("path", "sha256", "size_bytes")
    ):
        raise FastlaneReviewError("fastlane strict failure lineage is missing")
    if re.fullmatch(r"[0-9a-f]{64}", str(strict_failure["sha256"])) is None:
        raise FastlaneReviewError("fastlane strict failure hash is invalid")
    actions = payload.get("actions")
    advisories = payload.get("edge_stretch_advisories")
    expected = set(renderer.MOTIONS.values())
    if (
        not isinstance(actions, Mapping)
        or set(actions) != expected
        or not isinstance(advisories, Mapping)
        or set(advisories) != expected
    ):
        raise FastlaneReviewError("fastlane Walk/Idle action inventory is not exact")
    projected = copy.deepcopy(dict(payload))
    for action_name in sorted(expected):
        action = actions[action_name]
        advisory = advisories[action_name]
        if (
            not isinstance(action, Mapping)
            or action.get("action_name") != action_name
            or action.get("status")
            != "fastlane_hard_gates_passed_pending_visual_qa"
            or not isinstance(advisory, Mapping)
            or advisory.get("policy") != EDGE_POLICY
            or advisory.get("strict_formal_limit_exceeded") is not True
            or advisory.get("formal_registration_authorized") is not False
        ):
            raise FastlaneReviewError(f"fastlane action/advisory is invalid: {action_name}")
        projected["actions"][action_name]["status"] = "passed"
    return projected


def decorate_review_manifest(
    payload: Mapping[str, Any], *, source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if payload.get("schema") != renderer.REVIEW_MANIFEST_SCHEMA:
        raise FastlaneReviewError("strict renderer did not build its pinned manifest")
    if not isinstance(source_manifest, Mapping) or any(
        not source_manifest.get(field) for field in ("path", "sha256", "size_bytes")
    ):
        raise FastlaneReviewError("source fastlane retarget record is incomplete")
    result = copy.deepcopy(dict(payload))
    result.update(
        {
            "execution_track": TRACK,
            "formal_dataset_asset": False,
            "fastlane_visual_gate_status": "rendered_pending_agent_visual_qa",
            "source_fastlane_retarget_manifest": dict(source_manifest),
            "fastlane_review_wrapper": renderer.file_record(Path(__file__).resolve()),
        }
    )
    return result


def run_fastlane_review(args: argparse.Namespace) -> Path:
    retarget_path = Path(os.path.abspath(os.fspath(args.retarget_manifest)))
    source_record = renderer.file_record(retarget_path)
    strict_load = renderer._load_json
    strict_builder = renderer.build_review_manifest

    def fastlane_load(path: Path, description: str) -> dict[str, Any]:
        payload = strict_load(path, description)
        if description == "retarget manifest":
            return project_fastlane_manifest_for_strict_renderer(payload)
        return payload

    def fastlane_builder(**kwargs: Any) -> dict[str, Any]:
        return decorate_review_manifest(
            strict_builder(**kwargs), source_manifest=source_record
        )

    renderer._load_json = fastlane_load
    renderer.build_review_manifest = fastlane_builder
    try:
        return renderer.run_review_render(
            asset_id=args.asset_id,
            display_label=args.display_label,
            instance_kind=args.instance_kind,
            static_qa_json=args.static_qa_json,
            retarget_manifest=args.retarget_manifest,
            walking_glb=args.walking_glb,
            standing_idle_glb=args.standing_idle_glb,
            output_dir=args.output_dir,
            command=list(sys.argv),
        )
    finally:
        renderer._load_json = strict_load
        renderer.build_review_manifest = strict_builder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--display-label", required=True)
    parser.add_argument("--instance-kind", required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--retarget-manifest", type=Path, required=True)
    parser.add_argument("--walking-glb", type=Path, required=True)
    parser.add_argument("--standing-idle-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run_fastlane_review(parse_args(argv))
    print(f"TOKENRIG_FASTLANE_DYNAMIC_REVIEW_V1_OK {manifest}")
    return 0


if __name__ == "__main__":
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(blender_args))
