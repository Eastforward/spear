"""Estimate a generated animal's forward yaw from its rigged GLB.

Thin Blender wrapper over ``tools/quadruped_forward_estimation.py``.  Imports
the unanimated post-TokenRig GLB, collects world-space vertices of the single
skinned body mesh and writes the deterministic PCA + head-end-vote estimate as
draft review support.  The result is explicitly not review authority: a human
confirms the head end (directly or via a calibrated visual pre-screener)
before the value enters a forward declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import bpy
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.quadruped_forward_estimation import estimate_forward  # noqa: E402


SCHEMA = "avengine_generated_animal_forward_estimate_run_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_argv()
    source = args.input.resolve()
    output = args.output.resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input GLB: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    if not meshes:
        raise SystemExit("input contains no mesh")
    body = max(meshes, key=lambda item: len(item.data.vertices))
    matrix = np.asarray(body.matrix_world, dtype=np.float64)
    local = np.asarray(
        [vertex.co[:] for vertex in body.data.vertices], dtype=np.float64
    )
    world = local @ matrix[:3, :3].T + matrix[:3, 3]

    estimate = estimate_forward(world)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "estimate_pending_human_confirmation",
        "formal_dataset_registration_authorized": False,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "body_mesh": body.name,
        "vertex_count": int(len(world)),
        "estimate": estimate,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_FORWARD_ESTIMATE_OK "
        f"estimated_front_yaw_deg={estimate['estimated_front_yaw_deg']:.3f} "
        f"confidence={estimate['head_end_vote']['confidence']:.2f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
