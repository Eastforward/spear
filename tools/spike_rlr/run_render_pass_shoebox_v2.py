"""SPEAR/UE render pass for shoebox_v2 (5.2 x 6.0 x 2.8m + sofa @ (2.6, 3.45, 0.45)).

Piggybacks on the existing GPURIR pipeline's runtime scene spawner
(_spawn_shoebox in run_render_pass.py) which already builds 6 walls at
runtime via `spawn_room_piece`. We add ONE extra AStaticMeshActor for the
sofa, using the CubeMesh + a fabric material.

This is the "Task 3" full-automation of shoebox_v2: no manual UE editor
step, no .umap authoring — the UE Level itself stays whatever debug_0000
is; we spawn everything at runtime via SPEAR RPC.

Must be run under spear-env (has spear_ext + SPEAR RPC bindings) with
DISPLAY=:99 + VK_ICD_FILENAMES set. See run_all.sh for env setup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"

sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def _load_spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


def _spawn_sofa(game, spec):
    """Spawn the sofa as a StaticMeshActor at the SSOT-declared position.

    Uses UE Engine's basic CubeMesh + a wall material as the visual proxy;
    swap to a real sofa mesh later if needed for aesthetics.
    """
    from render_in_gpurir_room import resolve_wall_material, CUBE_MESH, M2CM

    sofa = spec["furniture"][0]  # only 1 furniture in v2 spec
    assert sofa["name"] == "sofa" and sofa["shape"] == "box"

    center_m = sofa["center_m"]      # (x, y, z_center) meters
    size_m = sofa["size_m"]           # (dx, dy, dz) meters
    # For a UE Cube (1m default), scale must convert to piece size.
    # Cube default is 100 cm; scale=1 -> 100 cm. So scale=size_m works
    # directly (piece[scale] is a multiplier of the 1m default cube).
    piece = {
        "name": "sofa",
        # UE cm coords; SSOT frame is right-handed Y-up meters. In the shoebox
        # runtime spawn, X/Y are directly mapped world->UE (no flip); Z is
        # actor center height (cube pivot is center of mesh by default).
        "location_cm": [center_m[0] * M2CM, center_m[1] * M2CM, center_m[2] * M2CM],
        "scale": [size_m[0], size_m[1], size_m[2]],  # 1m cube * scale
        "mesh": CUBE_MESH,
    }

    # Use wall material for now (fabric material asset would be nicer -- TODO)
    fabric_mat = resolve_wall_material(wall_material=None, wall_material_seed=1)

    from render_in_gpurir_room import spawn_room_piece
    actor = spawn_room_piece(game=game, piece=piece, material_path=fabric_mat,
                              cast_shadow=True)
    print(f"[shoebox_v2] spawned sofa at cm={piece['location_cm']} "
          f"scale={piece['scale']} material={fabric_mat}")
    return actor


def render_shoebox_v2(out_dir):
    """Run SPEAR RPC render pass on shoebox_v2 scene."""
    spec = _load_spec()

    # Monkey-patch _spawn_shoebox in run_render_pass to (a) use v2 room size
    # (5.2 x 6.0 x 2.8) and (b) additionally spawn our sofa. This keeps the
    # existing pipeline (view rendering, muxing, ...) as-is.
    from gpurir_scenes import run_render_pass as R
    from gpurir_scenes.run_render_pass import _spawn_shoebox as _orig_spawn

    def _spawn_shoebox_v2(game, room_size_m):
        # Override room_size with v2 dimensions (5.2 x 6.0 x 2.8)
        v2_size = tuple(spec["room_size_m"])
        print(f"[shoebox_v2] overriding room size {room_size_m} -> {v2_size}")
        _orig_spawn(game, v2_size)
        _spawn_sofa(game, spec)

    R._spawn_shoebox = _spawn_shoebox_v2

    # Also patch ROOM_SIZE_M constant in scene_spec (used by camera/mic
    # placement math)
    import gpurir_scenes.scene_spec as SS
    SS.ROOM_SIZE_M = tuple(spec["room_size_m"])

    # Now compose the scene using shoebox_v2 spec and hand-crafted trajectories
    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    from scene_two_dogs_v2 import compose_two_dog_scene_v2
    scene = compose_two_dog_scene_v2(SPEC_PATH)

    # Render
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R.run_render_pass(scene, "shoebox", str(out_dir))
    print(f"[shoebox_v2] rendered to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" / "spike_output" / "videos" / "A_gpurir_ue"))
    args = ap.parse_args()
    render_shoebox_v2(args.out)


if __name__ == "__main__":
    main()
