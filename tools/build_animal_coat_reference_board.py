#!/usr/bin/env python3
"""Build one deterministic square real-photo board for an animal coat edit.

The caller curates and rights-checks the input photographs.  This utility only
normalizes their presentation; it does not infer breed identity or licensing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


SCHEMA = "avengine_animal_coat_reference_board_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=3)
    return parser.parse_args()


def partition(total: int, count: int) -> list[int]:
    return [round(index * total / count) for index in range(count + 1)]


def main() -> int:
    args = parse_args()
    if not 4 <= len(args.input) <= 9:
        raise RuntimeError("reference board requires 4 to 9 curated photographs")
    if args.size not in {512, 1024, 1536, 2048}:
        raise RuntimeError("--size must be 512, 1024, 1536, or 2048")
    if not 2 <= args.columns <= 3:
        raise RuntimeError("--columns must be 2 or 3")
    output = args.output.resolve()
    if output.suffix.lower() != ".png":
        raise RuntimeError("reference board output must be PNG")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace reference board: {output}")
    inputs = [path.resolve() for path in args.input]
    for path in inputs:
        if not path.is_file():
            raise RuntimeError(f"missing reference photograph: {path}")

    rows = math.ceil(len(inputs) / args.columns)
    x_edges = partition(args.size, args.columns)
    y_edges = partition(args.size, rows)
    board = Image.new("RGB", (args.size, args.size), (242, 242, 242))
    for index, path in enumerate(inputs):
        column = index % args.columns
        row = index // args.columns
        width = x_edges[column + 1] - x_edges[column]
        height = y_edges[row + 1] - y_edges[row]
        with Image.open(path) as opened:
            opened.load()
            tile = ImageOps.fit(
                opened.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        board.paste(tile, (x_edges[column], y_edges[row]))

    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, format="PNG", optimize=False, compress_level=6)
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "board": str(output),
        "inputs": [str(path) for path in inputs],
        "image_count": len(inputs),
        "layout": {"columns": args.columns, "rows": rows},
        "resolution": [args.size, args.size],
        "source_files_copied": False,
        "breed_identity_and_rights_must_be_reviewed_separately": True,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ANIMAL_COAT_REFERENCE_BOARD_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
