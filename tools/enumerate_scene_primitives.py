"""Diagnostic: after loading Entry map, spawn our GPURIR room + animals,
then enumerate every actor and every UPrimitiveComponent under it. Prints
class name, source asset (mesh / material), and world-space bounding box.

Goal: find the "mystery checkerboard cube" that keeps appearing on the east
wall of the GPURIR room even after we destroy all StaticMesh/Skeletal/Brush/
Decal actors.
"""

import math
import os
import sys


_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../examples"
sys.path.insert(0, os.path.abspath(_EXAMPLES_DIR))

from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance,
    spawn_room_piece,
    spawn_directional_light,
    spawn_sky,
    spawn_point_light,
    compute_shoebox_room_layout,
    FLOOR_MATERIAL,
    WALL_MATERIAL,
    M2CM,
)


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        # 1) Spawn ONLY the 6 shoebox walls + 2 lights (nothing else).
        with instance.begin_frame():
            for p in compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8)):
                spawn_room_piece(
                    game=game,
                    piece=p,
                    material_path=(FLOOR_MATERIAL if p["name"] == "floor" else WALL_MATERIAL),
                )
            spawn_sky(game=game)
            spawn_directional_light(
                game=game, yaw_deg=-90.0, pitch_deg=-40.0, intensity_lux=10.0
            )
            spawn_point_light(
                game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0,
                intensity_lumens=2200.0, attenuation_cm=600.0,
            )
        with instance.end_frame():
            pass

        # 2) Enumerate every actor.
        with instance.begin_frame():
            actors = game.unreal_service.find_actors_by_class(uclass="AActor")
            print(f"\n==== TOTAL ACTORS: {len(actors)} ====", flush=True)
            for i, actor in enumerate(actors):
                # actor stable name
                try:
                    stable = game.unreal_service.get_stable_name_for_actor(
                        actor=actor, include_unreal_name=True
                    )
                except Exception:
                    stable = "<no-stable-name>"
                # world location
                try:
                    loc = actor.K2_GetActorLocation(as_dict=True)
                    loc_s = f"({loc['x']:.0f},{loc['y']:.0f},{loc['z']:.0f})"
                except Exception:
                    loc_s = "<no-loc>"
                # actor bounds
                try:
                    bounds = actor.GetActorBounds(
                        bOnlyCollidingComponents=False, as_dict=True
                    )
                    origin = bounds["Origin"]
                    extent = bounds["BoxExtent"]
                    bnd_s = (
                        f" bounds_orig=({origin['x']:.0f},{origin['y']:.0f},{origin['z']:.0f})"
                        f" bounds_ext=({extent['x']:.0f},{extent['y']:.0f},{extent['z']:.0f})"
                    )
                except Exception:
                    bnd_s = ""
                print(f"[{i:03d}] {stable}  loc={loc_s}{bnd_s}", flush=True)
        with instance.end_frame():
            pass
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
