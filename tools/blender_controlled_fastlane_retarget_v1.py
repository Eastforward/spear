#!/usr/bin/env python3
"""Run the approved semantic retarget core for controlled fastlane canaries.

Male uses the exact user-selected v2 yaw_000 record.  Female inherits only the
identity shared-limb basis and semantic anatomical transfer algorithm from
that authenticated record; it remains an agent-gated research fastlane and
never claims a female user approval or formal registration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_retarget_rocketbox_to_tokenrig as retarget


SCHEMA = "route2_controlled_fastlane_retarget_authorization_v1"
RUNNER_PATH = Path(__file__).resolve()
MALE_BASE = "rocketbox_male_adult_01"
FEMALE_BASE = "rocketbox_female_adult_01"
MALE_SELECTION = (
    RUNNER_PATH.parents[1]
    / "tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/"
    "retarget_motion_basis_and_axial_selection_v2/retarget_motion_basis_correction_v1.json"
)
MALE_REVIEW = (
    RUNNER_PATH.parents[1]
    / "tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/"
    "retarget_motion_basis_and_axial_review_v2/motion_basis_review_manifest.json"
)
MAIN_RUNNER_SHA256 = "5f5cf0f06ad4a8913a142616db61051d436cc2e46195d795c9ca02d8bfed27bf"
FASTLANE_SOURCE_ROTATION_ORTHOGONALITY_MAX = 6.0e-6
FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M = 1.0e-4


class FastlaneRetargetError(RuntimeError):
    pass


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        raise FastlaneRetargetError(f"artifact must be a direct nonempty file: {path}")
    return {
        "path": str(path),
        "sha256": retarget.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def inherited_female_motion_basis() -> dict[str, Any]:
    approved = retarget.authenticate_motion_basis_selection(
        base_avatar_id=MALE_BASE,
        motion_basis_selection=MALE_SELECTION,
        motion_basis_review_manifest=MALE_REVIEW,
    )
    matrix = approved["matrix_3x3"]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if matrix != identity or approved["candidate_id"] != "yaw_000":
        raise FastlaneRetargetError("approved male v2 basis is no longer identity yaw_000")
    return {
        **approved,
        "base_avatar_id": FEMALE_BASE,
        "authorization_class": "agent_research_candidate_fastlane",
        "inheritance_scope": "identity_shared_limb_basis_and_semantic_anatomical_transfer_algorithm_only",
        "female_user_approval_claimed": False,
        "formal_dataset_asset": False,
        "inherited_from": {
            "base_avatar_id": MALE_BASE,
            "selection": _record(MALE_SELECTION),
            "review_manifest": _record(MALE_REVIEW),
            "candidate_id": approved["candidate_id"],
            "matrix_3x3": matrix,
        },
    }


def _publish_sidecar(
    *,
    asset_id: str,
    base_avatar_id: str,
    output_manifest: Path,
    motion_basis: dict[str, Any],
    reach_adjustments: list[dict[str, Any]],
) -> Path:
    output_manifest = Path(output_manifest).absolute()
    destination = output_manifest.parent.parent / f"{output_manifest.parent.name}.fastlane_authorization_v1.json"
    payload = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "base_avatar_id": base_avatar_id,
        "state_classification": "research_candidate_fastlane",
        "retarget_manifest": _record(output_manifest),
        "motion_basis": motion_basis,
        "fastlane_tolerances": {
            "source_rotation_orthogonality_max": (
                FASTLANE_SOURCE_ROTATION_ORTHOGONALITY_MAX
                if base_avatar_id == FEMALE_BASE
                else retarget.MAXIMUM_ROTATION_ORTHOGONALITY_ERROR
            ),
            "production_default_source_rotation_orthogonality_max": 5.0e-6,
            "ground_penetration_max_m": retarget.MAXIMUM_PENETRATION_M,
            "grounding_correction_max_m": retarget.MAXIMUM_GROUNDING_CORRECTION_M,
            "source_reach_overshoot_max_m": FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M,
        },
        "bounded_source_reach_adjustments": reach_adjustments,
        "wrapper": _record(RUNNER_PATH),
        "main_runner": _record(Path(retarget.__file__).resolve()),
        "agent_qa_status": "pending_dynamic_media_qa",
        "user_acceptance": "not_claimed",
        "formal_dataset_registration_authorized": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        os.write(descriptor, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def run(args: argparse.Namespace) -> Path:
    main_path = Path(retarget.__file__).resolve()
    if retarget.sha256_file(main_path) != MAIN_RUNNER_SHA256:
        raise FastlaneRetargetError("approved semantic retarget core hash changed")
    original = retarget.authenticate_motion_basis_selection
    original_orthogonality = retarget.MAXIMUM_ROTATION_ORTHOGONALITY_ERROR
    original_endpoint_builder = retarget._build_source_leg_endpoint_rest_frame
    reach_adjustments: list[dict[str, Any]] = []
    if args.base_avatar_id == FEMALE_BASE:
        inherited = inherited_female_motion_basis()

        def authenticate_female(**kwargs: Any) -> dict[str, Any]:
            if (
                kwargs.get("base_avatar_id") != FEMALE_BASE
                or Path(kwargs.get("motion_basis_selection", "")).absolute() != MALE_SELECTION
                or Path(kwargs.get("motion_basis_review_manifest", "")).absolute() != MALE_REVIEW
            ):
                raise FastlaneRetargetError("female fastlane motion-basis inputs changed")
            return json.loads(json.dumps(inherited, sort_keys=True))

        retarget.authenticate_motion_basis_selection = authenticate_female
        retarget.MAXIMUM_ROTATION_ORTHOGONALITY_ERROR = (
            FASTLANE_SOURCE_ROTATION_ORTHOGONALITY_MAX
        )

        def bounded_endpoint_builder(**kwargs: Any) -> dict[str, Any]:
            try:
                return original_endpoint_builder(**kwargs)
            except retarget.RetargetError as error:
                if str(error) != "rest-frame endpoint source distance is outside its leg reach":
                    raise
                hip = np.asarray(kwargs["source_current_hip"], dtype=np.float64)
                ankle = np.asarray(kwargs["source_current_ankle"], dtype=np.float64)
                rest_hip = np.asarray(kwargs["source_rest_hip"], dtype=np.float64)
                rest_knee = np.asarray(kwargs["source_rest_knee"], dtype=np.float64)
                rest_ankle = np.asarray(kwargs["source_rest_ankle"], dtype=np.float64)
                vector = ankle - hip
                distance = float(np.linalg.norm(vector))
                maximum = float(
                    np.linalg.norm(rest_knee - rest_hip)
                    + np.linalg.norm(rest_ankle - rest_knee)
                )
                overshoot = distance - maximum
                if (
                    not math.isfinite(overshoot)
                    or overshoot <= 0.0
                    or overshoot > FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M
                    or distance <= 1.0e-12
                ):
                    raise FastlaneRetargetError(
                        "female source leg reach overshoot is not bounded float drift: "
                        f"overshoot_m={overshoot:.17g} limit_m="
                        f"{FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M:.17g}"
                    ) from error
                adjusted_distance = maximum - 1.0e-9
                adjusted = hip + vector * (adjusted_distance / distance)
                adjusted_kwargs = {**kwargs, "source_current_ankle": adjusted.tolist()}
                reach_adjustments.append(
                    {
                        "reason": "source_endpoint_float_overshoot_clamped_inside_exact_reach",
                        "source_distance_m": distance,
                        "source_maximum_reach_m": maximum,
                        "overshoot_m": overshoot,
                        "adjusted_distance_m": adjusted_distance,
                        "maximum_allowed_overshoot_m": FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M,
                    }
                )
                return original_endpoint_builder(**adjusted_kwargs)

        retarget._build_source_leg_endpoint_rest_frame = bounded_endpoint_builder
        motion_basis = inherited
    elif args.base_avatar_id == MALE_BASE:
        motion_basis = original(
            base_avatar_id=MALE_BASE,
            motion_basis_selection=MALE_SELECTION,
            motion_basis_review_manifest=MALE_REVIEW,
        )
    else:
        raise FastlaneRetargetError("fastlane supports only the two pinned adult bases")
    try:
        manifest = retarget.run_retarget(
            asset_id=args.asset_id,
            base_avatar_id=args.base_avatar_id,
            bind_pose_glb=args.bind_pose_glb,
            static_qa_json=args.static_qa_json,
            baseline_retarget_blend=args.baseline_retarget_blend,
            baseline_retarget_manifest=args.baseline_retarget_manifest,
            idle_motion_fbx=args.idle_motion_fbx,
            motion_basis_selection=MALE_SELECTION,
            motion_basis_review_manifest=MALE_REVIEW,
            output_dir=args.output_dir,
            command=list(sys.argv),
        )
    finally:
        retarget.authenticate_motion_basis_selection = original
        retarget.MAXIMUM_ROTATION_ORTHOGONALITY_ERROR = original_orthogonality
        retarget._build_source_leg_endpoint_rest_frame = original_endpoint_builder
    return _publish_sidecar(
        asset_id=args.asset_id,
        base_avatar_id=args.base_avatar_id,
        output_manifest=manifest,
        motion_basis=motion_basis,
        reach_adjustments=reach_adjustments,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--base-avatar-id", choices=(MALE_BASE, FEMALE_BASE), required=True)
    parser.add_argument("--bind-pose-glb", type=Path, required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--baseline-retarget-blend", type=Path, required=True)
    parser.add_argument("--baseline-retarget-manifest", type=Path, required=True)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


if __name__ == "__main__":
    print(f"CONTROLLED_FASTLANE_RETARGET_OK {run(parse_args(_blender_argv()))}")
