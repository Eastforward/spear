"""Symmetric counterpart to tools/gpurir_scenes/dump_apartment_furniture.py.

Dumps only shell (structural) actors from apartment_0000: walls, floor,
ceiling, doors, windows, curtains, pictures, mirrors, and misc large
structural meshes. Output goes to data/apartment_shell_map.json for
downstream RLR mesh construction (tools/spike_rlr/gen_mesh_apartment.py).

Usage:
    export DISPLAY=:99
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/spike_rlr/dump_apartment_shell.py --dry-run
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/spike_rlr/dump_apartment_shell.py --out data/apartment_shell_map.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "gpurir_scenes"))

from render_in_apartment import APARTMENT_MAP, configure_instance  # noqa: E402
from apartment_actor_classifier import classify_actor, SHELL_LABELS  # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def dump_apartment_shell(out_path, dry_run=False):
    instance = configure_instance(rpc_port=39003)
    game = instance.get_game()

    try:
        with instance.begin_frame():
            actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
            print(f"[shell] found {len(actors)} AStaticMeshActor instances", flush=True)

            records = []
            label_counts = {lbl: 0 for lbl in SHELL_LABELS}
            label_counts["furniture"] = 0

            for actor in actors:
                try:
                    stable = game.unreal_service.get_stable_name_for_actor(
                        actor=actor, include_unreal_name=True
                    )
                except Exception:
                    stable = "<no-stable-name>"
                if not stable:
                    stable = "<empty-name>"

                try:
                    b = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
                    origin = b["Origin"]
                    ext = b["BoxExtent"]
                except Exception:
                    continue  # skip actors with no valid bounds

                bbox_min = [origin["x"] - ext["x"], origin["y"] - ext["y"], origin["z"] - ext["z"]]
                bbox_max = [origin["x"] + ext["x"], origin["y"] + ext["y"], origin["z"] + ext["z"]]

                try:
                    loc = actor.K2_GetActorLocation(as_dict=True)
                    loc_list = [loc["x"], loc["y"], loc["z"]]
                except Exception:
                    loc_list = [origin["x"], origin["y"], origin["z"]]

                try:
                    rot = actor.K2_GetActorRotation(as_dict=True)
                    rot_list = [rot["roll"], rot["pitch"], rot["yaw"]]
                except Exception:
                    rot_list = [0.0, 0.0, 0.0]

                x_ext = ext["x"] * 2.0
                y_ext = ext["y"] * 2.0

                label = classify_actor(stable, bbox_min[2], bbox_max[2], x_ext, y_ext)
                label_counts[label] = label_counts.get(label, 0) + 1

                # Keep only shell actors (inverted filter vs the furniture dump)
                if label not in SHELL_LABELS:
                    continue

                records.append({
                    "actor_name": stable,
                    "shell_label": label,
                    "uclass": "AStaticMeshActor",
                    "bbox_min_ue_cm": bbox_min,
                    "bbox_max_ue_cm": bbox_max,
                    "actor_location_ue_cm": loc_list,
                    "actor_rotation_deg": rot_list,
                })
        with instance.end_frame():
            pass

        print(f"\n[shell] label counts: {label_counts}", flush=True)
        print(f"[shell] {len(records)} shell actors kept "
              f"(of {len(actors)} total)", flush=True)

        if dry_run:
            print("\n[shell] --dry-run: not writing JSON", flush=True)
            return

        if out_path is None:
            raise SystemExit("[shell] --out is required unless --dry-run")

        payload = {
            "meta": {
                "apartment_map_path": APARTMENT_MAP,
                "dump_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "spear_commit": _git_head(),
                "ue_version": "5.5",
                "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
                "num_actors_seen": len(actors),
                "num_actors_after_filter": len(records),
                "shell_label_counts": label_counts,
            },
            "shell_actors": records,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[shell] wrote {out_path}", flush=True)

    finally:
        instance.close(force=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="Output JSON path. Required unless --dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print label counts only, don't write JSON.")
    args = p.parse_args()
    dump_apartment_shell(out_path=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
