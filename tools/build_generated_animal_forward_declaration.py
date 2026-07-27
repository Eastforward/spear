#!/usr/bin/env python3
"""Author the single authoritative forward declaration for a generated animal.

Combines the deterministic estimator output with the human head-end
confirmation into one self-hashed declaration.  Every downstream stage
(heading normalization, retarget, UE binding) derives the anatomical front
from this file; nothing else may re-declare it.

Typical flow:
  1. blender_estimate_generated_animal_forward.py writes the estimate JSON.
  2. A human confirms which candidate end is the head (recording evidence,
     e.g. the reviewed candidate render or review page decision).
  3. This tool binds estimate + confirmation into the declaration.

If the confirmed yaw disagrees with the estimate by more than the flip
tolerance around either candidate, the build fails: that means the human and
the estimator are not looking at the same asset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_animal_forward_contract import (  # noqa: E402
    ForwardContractError,
    build_forward_declaration,
)


CANDIDATE_MATCH_TOLERANCE_DEG = 15.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-workspace", required=True)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--estimate-json", type=Path)
    parser.add_argument("--confirmed-front-yaw-deg", type=float, required=True)
    parser.add_argument("--head-end-evidence", type=Path, required=True)
    parser.add_argument(
        "--head-end-decision-source",
        choices=("human_review", "human_confirming_estimator"),
        required=True,
    )
    parser.add_argument(
        "--target-species",
        choices=("dog", "cat"),
        default="dog",
        help=(
            "Generated target species. Selects the pinned species motion donor; "
            "the donor supplies skeleton semantics and actions, never body geometry."
        ),
    )
    parser.add_argument(
        "--motion-donor-tag",
        help=(
            "Optional explicit donor ID. It must match --target-species; omit "
            "to use the registered species donor."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace declaration: {output}")

    estimate = None
    if args.estimate_json is not None:
        estimate_path = args.estimate_json.resolve()
        if (
            estimate_path.is_symlink()
            or not estimate_path.is_file()
            or estimate_path.stat().st_size <= 0
        ):
            raise SystemExit(f"missing or unsafe estimate JSON: {estimate_path}")
        run = json.loads(estimate_path.read_text(encoding="utf-8"))
        estimate = run.get("estimate")
        if not isinstance(estimate, dict):
            raise SystemExit("estimate JSON does not contain an estimate object")
        candidates = estimate.get("candidate_front_yaw_degrees", [])
        if not any(
            angular_difference(float(candidate), args.confirmed_front_yaw_deg)
            <= CANDIDATE_MATCH_TOLERANCE_DEG
            for candidate in candidates
        ):
            raise SystemExit(
                "confirmed yaw matches neither estimator candidate "
                f"(candidates={candidates}, confirmed="
                f"{args.confirmed_front_yaw_deg}); the human and the "
                "estimator are not reviewing the same asset"
            )
        if args.head_end_decision_source != "human_confirming_estimator":
            raise SystemExit(
                "an estimate was provided; use "
                "--head-end-decision-source human_confirming_estimator"
            )
        estimate = {**estimate, "estimate_run_path": str(estimate_path)}
    elif args.head_end_decision_source != "human_review":
        raise SystemExit(
            "without an estimate JSON the decision source must be human_review"
        )

    try:
        declaration = build_forward_declaration(
            asset_workspace=args.asset_workspace,
            input_glb=args.input_glb,
            reviewed_source_front_yaw_deg=args.confirmed_front_yaw_deg,
            head_end_decision_source=args.head_end_decision_source,
            head_end_evidence=args.head_end_evidence,
            motion_donor_tag=args.motion_donor_tag,
            target_species=args.target_species,
            estimate=estimate,
        )
    except ForwardContractError as error:
        raise SystemExit(f"forward declaration build failed: {error}") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(declaration, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_FORWARD_DECLARATION_OK "
        f"reviewed_source_front_yaw_deg={args.confirmed_front_yaw_deg:.3f} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
