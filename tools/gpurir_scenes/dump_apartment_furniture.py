"""One-shot offline dump of apartment_0000 static-mesh actor bboxes.

Runs SPEAR RPC, enumerates every AStaticMeshActor in the apartment,
applies three-layer filter (z-height / name-keyword / bbox-area) to
remove walls/floor/ceiling, writes surviving bboxes as UE cm coords to
data/apartment_furniture_map.json for downstream collision checks.

Usage:
    export DISPLAY=:99
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/gpurir_scenes/dump_apartment_furniture.py --dry-run
    # review z-histogram and filter reasons, then:
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/gpurir_scenes/dump_apartment_furniture.py --out data/apartment_furniture_map.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = "/data/jzy/code/SPEAR"
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from render_in_apartment import APARTMENT_MAP, configure_instance  # noqa: E402


# Filter thresholds (UE world Z is from apartment origin, NOT room floor).
# Apartment floor ~27 cm; ceiling ~280 cm. Thresholds are STARTING values.
# After --dry-run, review the z-histogram and adjust if needed.
Z_CEILING_CM = 300.0     # bbox_min_z > this -> drop (ceiling)
Z_FLOOR_CM = 5.0         # bbox_max_z < this -> drop (floor patch/decal)
NAME_KEYWORDS = ("wall", "floor", "ceiling", "ground")
BBOX_AREA_MAX_CM2 = 200000.0  # x_extent * y_extent > 20 m^2 -> drop (structural mesh)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _classify(actor_name: str, bbox_min_z: float, bbox_max_z: float,
              x_extent_cm: float, y_extent_cm: float) -> str:
    """Return filter reason string ("z_ceiling" / "name_wall" / ... / "kept")."""
    if bbox_min_z > Z_CEILING_CM:
        return "z_ceiling"
    if bbox_max_z < Z_FLOOR_CM:
        return "z_floor"
    nl = actor_name.lower()
    for kw in NAME_KEYWORDS:
        if kw in nl:
            return f"name_{kw}"
    if x_extent_cm * y_extent_cm > BBOX_AREA_MAX_CM2:
        return "bbox_too_large"
    return "kept"


def dump_apartment(out_path, dry_run=False):
    instance = configure_instance(rpc_port=39002)
    game = instance.get_game()

    from render_in_apartment import APARTMENT_MAP as MAP_PATH

    try:
        with instance.begin_frame():
            actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
            print(f"[dump] found {len(actors)} AStaticMeshActor instances", flush=True)

            records = []
            reasons = {"z_ceiling": 0, "z_floor": 0, "name_wall": 0,
                       "name_floor": 0, "name_ceiling": 0, "name_ground": 0,
                       "bbox_too_large": 0, "kept": 0}
            z_bins = [0] * 12  # 0-50, 50-100, ..., 550+ cm

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
                    continue  # actor with no valid bounds -- skip silently

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
                bin_idx = min(max(int(bbox_min[2] / 50), 0), 11)
                z_bins[bin_idx] += 1

                reason = _classify(stable, bbox_min[2], bbox_max[2], x_ext, y_ext)
                reasons[reason] += 1
                if reason != "kept":
                    continue

                records.append({
                    "actor_name": stable,
                    "uclass": "AStaticMeshActor",
                    "bbox_min_ue_cm": bbox_min,
                    "bbox_max_ue_cm": bbox_max,
                    "actor_location_ue_cm": loc_list,
                    "actor_rotation_deg": rot_list,
                })
        with instance.end_frame():
            pass

        print("\n[dump] z-histogram (bbox_min_z, 50 cm bins):", flush=True)
        for i, c in enumerate(z_bins):
            lo = i * 50
            hi = (i + 1) * 50 if i < 11 else "+"
            print(f"  {lo:4d}-{str(hi):>4} cm: {c}", flush=True)
        print(f"\n[dump] filter reasons: {reasons}", flush=True)
        print(f"[dump] kept {len(records)} / {len(actors)} actors", flush=True)

        # Guard: if any single reason >50% of total, warn (spec 4.1)
        total = sum(reasons.values())
        for r, c in reasons.items():
            if total > 0 and c > 0.5 * total and r != "kept":
                print(f"[dump] WARNING: reason={r} triggered on {c}/{total} "
                      f"(> 50%). Consider adjusting threshold before writing JSON.",
                      flush=True)

        if dry_run:
            print("\n[dump] --dry-run: not writing JSON", flush=True)
            return

        if out_path is None:
            raise SystemExit("[dump] --out is required unless --dry-run")

        payload = {
            "meta": {
                "apartment_map_path": MAP_PATH,
                "dump_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "spear_commit": _git_head(),
                "ue_version": "5.5",
                "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
                "num_actors_seen": len(actors),
                "num_actors_after_filter": len(records),
                "filter_reasons": reasons,
            },
            "furniture": records,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[dump] wrote {out_path}", flush=True)

    finally:
        instance.close(force=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="Output JSON path. Required unless --dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print z-histogram + filter reasons only, don't write JSON.")
    args = p.parse_args()
    dump_apartment(out_path=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
