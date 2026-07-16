#!/usr/bin/env python3
"""Run the Route-2 Walk/Idle research fastlane without weakening hard gates.

The production retarget runner remains unchanged.  This adapter records numeric
edge-stretch excess as an explicit visual-review advisory while delegating every
other action invariant to the strict validator.  Output is always a
``research_candidate_fastlane`` and never a formal dataset asset.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_retarget_rocketbox_to_tokenrig as runner


TRACK = "research_candidate_fastlane"
EDGE_POLICY = "recorded_numeric_advisory_requires_visual_tear_review_v1"
STRICT_FAILURE_SCHEMA = "tokenrig_rocketbox_retarget_attempt_v1"


def _edge_advisory(metrics: Mapping[str, Any]) -> dict[str, Any]:
    deformation = metrics.get("deformation")
    if not isinstance(deformation, Mapping):
        raise runner.RetargetError("fastlane action has no deformation evidence")
    actual = deformation.get("maximum_skinned_edge_stretch_ratio")
    allowed = deformation.get("allowed_maximum_skinned_edge_stretch_ratio")
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or float(actual) < 0.0
        or isinstance(allowed, bool)
        or not isinstance(allowed, (int, float))
        or not math.isfinite(float(allowed))
        or float(allowed) <= 0.0
    ):
        raise runner.RetargetError("fastlane edge-stretch evidence is invalid")
    return {
        "policy": EDGE_POLICY,
        "actual_maximum_ratio": float(actual),
        "strict_formal_limit_ratio": float(allowed),
        "strict_formal_limit_exceeded": float(actual) > float(allowed),
        "formal_registration_authorized": False,
    }


def validate_fastlane_action_metrics(
    metrics: Mapping[str, Any],
    *,
    semantic_mapping: Mapping[str, Any],
    strict_validator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Delegate all strict gates after isolating only the edge-ratio comparison."""

    advisory = _edge_advisory(metrics)
    delegated = copy.deepcopy(dict(metrics))
    deformation = delegated.get("deformation")
    if not isinstance(deformation, dict):
        raise runner.RetargetError("fastlane deformation evidence is not mutable JSON")
    if advisory["strict_formal_limit_exceeded"]:
        deformation["maximum_skinned_edge_stretch_ratio"] = advisory[
            "strict_formal_limit_ratio"
        ]
    result = strict_validator(delegated, semantic_mapping=semantic_mapping)
    if not isinstance(result, Mapping) or result.get("status") != "passed":
        raise runner.RetargetError("strict action validator did not return a pass")
    return {
        **dict(result),
        "status": "fastlane_hard_gates_passed_pending_visual_qa",
        "execution_track": TRACK,
        "edge_stretch_advisory": advisory,
    }


def decorate_fastlane_manifest(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    strict_failure: Mapping[str, Any],
) -> dict[str, Any]:
    actions = metrics.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(runner.ACTION_NAMES.values()):
        raise runner.RetargetError("fastlane metrics must contain exact Walk and Idle")
    advisories = {
        name: _edge_advisory(actions[name]) for name in sorted(actions)
    }
    if not isinstance(strict_failure, Mapping) or any(
        not strict_failure.get(field) for field in ("path", "sha256", "size_bytes")
    ):
        raise runner.RetargetError("strict failure evidence record is incomplete")
    result = copy.deepcopy(dict(payload))
    if (
        result.get("schema") != runner.MANIFEST_SCHEMA
        or result.get("state_classification") != "research_candidate"
        or result.get("automatic_checks") != "passed"
    ):
        raise runner.RetargetError("strict runner manifest is not a passed research candidate")
    result.update(
        {
            "execution_track": TRACK,
            "formal_dataset_asset": False,
            "automatic_check_scope": "fastlane_hard_gates_only",
            "strict_formal_registration_status": (
                "blocked_by_recorded_edge_stretch"
                if any(
                    record["strict_formal_limit_exceeded"]
                    for record in advisories.values()
                )
                else "not_blocked_by_edge_stretch"
            ),
            "edge_stretch_advisories": advisories,
            "strict_failure_evidence": dict(strict_failure),
            "fastlane_wrapper": runner.file_descriptor(Path(__file__).resolve()),
        }
    )
    return result


def authenticate_strict_failure(path: Path, *, asset_id: str) -> dict[str, Any]:
    value = Path(os.path.abspath(os.fspath(path)))
    if (
        value.is_symlink()
        or not value.is_file()
        or value.resolve() != value
        or value.name != "retarget_failure.json"
        or value.stat().st_mode & 0o777 != 0o444
    ):
        raise runner.RetargetError("strict failure must be immutable retarget_failure.json")
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise runner.RetargetError(f"strict failure evidence is invalid JSON: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != STRICT_FAILURE_SCHEMA
        or payload.get("asset_id") != asset_id
        or payload.get("decision") != "rejected"
        or payload.get("readiness_bundle_published") is not False
        or payload.get("error_type") != "RetargetError"
        or re.search(r"edge stretch", str(payload.get("error", "")), re.IGNORECASE)
        is None
    ):
        raise runner.RetargetError("strict failure is not the exact edge-stretch rejection")
    return runner.file_descriptor(value)


def run_fastlane(args: argparse.Namespace) -> Path:
    strict_failure = authenticate_strict_failure(
        args.strict_failure_evidence, asset_id=args.asset_id
    )
    strict_validator = runner.validate_action_metrics
    strict_manifest_builder = runner.build_retarget_manifest

    def fastlane_validator(
        metrics: Mapping[str, Any], *, semantic_mapping: Mapping[str, Any]
    ) -> dict[str, Any]:
        return validate_fastlane_action_metrics(
            metrics,
            semantic_mapping=semantic_mapping,
            strict_validator=strict_validator,
        )

    def fastlane_manifest_builder(**kwargs: Any) -> dict[str, Any]:
        payload = strict_manifest_builder(**kwargs)
        return decorate_fastlane_manifest(
            payload, kwargs["metrics"], strict_failure=strict_failure
        )

    runner.validate_action_metrics = fastlane_validator
    runner.build_retarget_manifest = fastlane_manifest_builder
    try:
        return runner.run_retarget(
            asset_id=args.asset_id,
            base_avatar_id=args.base_avatar_id,
            bind_pose_glb=args.bind_pose_glb,
            static_qa_json=args.static_qa_json,
            baseline_retarget_blend=args.baseline_retarget_blend,
            baseline_retarget_manifest=args.baseline_retarget_manifest,
            idle_motion_fbx=args.idle_motion_fbx,
            motion_basis_selection=args.motion_basis_selection,
            motion_basis_review_manifest=args.motion_basis_review_manifest,
            output_dir=args.output_dir,
            command=list(sys.argv),
        )
    finally:
        runner.validate_action_metrics = strict_validator
        runner.build_retarget_manifest = strict_manifest_builder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--base-avatar-id", required=True)
    parser.add_argument("--bind-pose-glb", type=Path, required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--baseline-retarget-blend", type=Path, required=True)
    parser.add_argument("--baseline-retarget-manifest", type=Path, required=True)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--motion-basis-selection", type=Path, required=True)
    parser.add_argument("--motion-basis-review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-failure-evidence", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_fastlane(args)
    print(f"TOKENRIG_RETARGET_FASTLANE_V1_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(blender_args))
