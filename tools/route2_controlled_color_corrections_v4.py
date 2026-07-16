#!/usr/bin/env python3
"""Regenerate the two rejected female color canaries with corrected masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import route2_controlled_color_references_v3 as base


RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
OUTPUT_ROOT = SPEAR_ROOT / "tmp/route2_controlled_color_corrections_v4"
PIXAL_OUTPUT_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_color_v4/pixal3d"
TOP_MASK = {
    "strategy": "reviewed_source_hair_polygons",
    "semantic": "female_short_sleeve_top_exact_polygons_v4",
    "polygons": [
        [[0.390, 0.225], [0.460, 0.215], [0.540, 0.215], [0.610, 0.225],
         [0.620, 0.490], [0.380, 0.490]],
        [[0.390, 0.225], [0.355, 0.238], [0.320, 0.275], [0.345, 0.315],
         [0.382, 0.335], [0.415, 0.252]],
        [[0.610, 0.225], [0.645, 0.238], [0.680, 0.275], [0.655, 0.315],
         [0.618, 0.335], [0.585, 0.252]],
    ],
    "radius": 3,
}
HAIR_MASK = {
    "strategy": "reviewed_source_hair_polygons",
    "semantic": "female_fixed_hair_excluding_face_v4",
    "polygons": [
        [[0.462, 0.119], [0.455, 0.108], [0.465, 0.097], [0.485, 0.091],
         [0.515, 0.091], [0.538, 0.101], [0.545, 0.114], [0.532, 0.120],
         [0.515, 0.108], [0.485, 0.108], [0.470, 0.119]],
        [[0.455, 0.108], [0.470, 0.112], [0.466, 0.148], [0.452, 0.153]],
        [[0.532, 0.110], [0.545, 0.113], [0.548, 0.154], [0.536, 0.148]],
        [[0.538, 0.145], [0.557, 0.150], [0.570, 0.180], [0.568, 0.235],
         [0.550, 0.242], [0.542, 0.205]],
    ],
    "radius": 3,
}
CASE_SPECS = (
    base._case(
        "female_top_teal_mask_v4",
        "female",
        "top_color",
        409,
        "muted teal",
        (40, 122, 120),
        TOP_MASK,
    ),
    base._case(
        "female_hair_chestnut_mask_v4",
        "female",
        "fixed_hair_color",
        410,
        "chestnut brown",
        (112, 69, 47),
        HAIR_MASK,
    ),
)
CASE_BY_ID = {case["case_id"]: case for case in CASE_SPECS}


def configure() -> None:
    base.SCHEMA = "route2_controlled_color_reference_jobs_v4"
    base.CANDIDATE_SCHEMA = "route2_controlled_color_reference_candidate_v4"
    base.DECISION_SCHEMA = "route2_controlled_color_reference_agent_qa_v4"
    base.PIXAL_JOBS_SCHEMA = "route2_controlled_color_pixal_jobs_v4"
    base.RUNNER_PATH = RUNNER_PATH
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PIXAL_OUTPUT_ROOT = PIXAL_OUTPUT_ROOT
    base.CASE_SPECS = CASE_SPECS
    base.CASE_BY_ID = CASE_BY_ID


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    generate = commands.add_parser("generate")
    generate.add_argument("--case-id", action="append", choices=tuple(CASE_BY_ID), required=True)
    generate.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure()
    if args.command == "prepare":
        print(f"ROUTE2_CONTROLLED_COLOR_V4_PREPARED {base.prepare()}")
    else:
        print(json.dumps(base.generate(args.case_id, args.gpu), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
