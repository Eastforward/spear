"""Fail-closed gait-direction audit for a retargeted quadruped GLB.

Thin Blender wrapper over ``tools/quadruped_gait_direction.py``.  Samples the
Walking action on the animated, heading-normalized GLB and verifies from
stance-foot drift that the animal locomotes head-first along the canonical
+X anatomical front.  Backward or sideways gaits — the historical
whole-batch human-rejection classes — exit non-zero so the outer pipeline
stops before rendering or human review time is spent.
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

from tools.quadruped_gait_direction import classify_gait_direction  # noqa: E402


SCHEMA = "avengine_generated_animal_gait_direction_run_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", default="Walking")
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluated_world_vertices(mesh_object, depsgraph):
    evaluated = mesh_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", local.ravel())
        return local @ matrix[:3, :3].T + matrix[:3, 3]
    finally:
        evaluated.to_mesh_clear()


def main():
    args = parse_argv()
    source = args.input.resolve()
    output = args.output.resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input GLB: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 3 <= args.samples <= 120:
        raise SystemExit("--samples must be in [3, 120]")
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if not meshes or len(armatures) != 1:
        raise SystemExit("input must contain a mesh and exactly one armature")
    body = max(meshes, key=lambda item: len(item.data.vertices))
    armature = armatures[0]
    action = next(
        (
            candidate
            for candidate in bpy.data.actions
            if args.action.lower() in candidate.name.lower()
        ),
        None,
    )
    if action is None:
        raise SystemExit(
            f"missing action {args.action}; "
            f"available={[item.name for item in bpy.data.actions]}"
        )

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    armature.data.pose_position = "POSE"
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    start, end = action.frame_range
    frames = []
    for frame in np.linspace(float(start), float(end), args.samples):
        scene.frame_set(int(round(float(frame))))
        bpy.context.view_layer.update()
        frames.append(evaluated_world_vertices(body, depsgraph))

    result = classify_gait_direction(frames)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_dataset_registration_authorized": False,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "action": action.name,
        "samples": args.samples,
        "result": result,
        "status": "pass" if result["walks_head_first"] else "fail",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not result["walks_head_first"]:
        raise SystemExit(
            "GAIT_DIRECTION_AUDIT_FAILED "
            f"classification={result['classification']} "
            f"stance_drift_yaw_deg={result['stance_drift_yaw_deg']:.2f} "
            f"output={output}"
        )
    print(
        "GAIT_DIRECTION_AUDIT_OK "
        f"classification={result['classification']} "
        f"stance_drift_yaw_deg={result['stance_drift_yaw_deg']:.2f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
