"""Diagnostic: spawn ONE animated dog centered in an empty room, park a
camera close to it in a side-on view, render a lot of frames with the dog
NOT translating (so any per-frame delta must come from anim tick), save
the frames and compute an honest 'is the pose actually changing' metric.

The metric: for each frame, threshold the dog silhouette (the dog is the
only non-floor/wall object), morph-open it a bit, and compute the number
of pixels that are 'newly foreground' or 'newly background' vs the
previous frame. If the anim is truly ticking, silhouette pixels flip
regularly (~1-5% of the silhouette area per frame) because legs / tail
change position. If the anim is frozen, silhouette flip is ~0 (only AA /
lighting jitter).

Output: /tmp/diag_dog/frame_XXXX.png, /tmp/diag_dog/report.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import spear

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))

from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance, spawn_directional_light, spawn_point_light, spawn_sky, M2CM,
)
from render_in_apartment import spawn_camera, read_frame, clean_frames  # noqa: E402

ANIMATED_DOG_BP = "/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C"
WALKING_ANIM = "/Game/MyAssets/Audioset/Meshes/animated_dog/Walking"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--per-frame-warmup", type=int, default=6,
                   help="how many sim frames to advance between renders")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--rpc-port", type=int, default=39002)
    p.add_argument("--dog-scale", type=float, default=0.12)
    p.add_argument("--output-dir", default="/tmp/diag_dog")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    clean_frames(args.output_dir)

    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # Purge default actors
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                        "AStaticMeshActor", "ASkeletalMeshActor",
                        "ABrush", "ADecalActor", "AInstancedFoliageActor"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            spawn_sky(game=game)
            spawn_directional_light(game=game, yaw_deg=-30.0, pitch_deg=-45.0, intensity_lux=8.0)
            spawn_point_light(game=game, x_cm=0.0, y_cm=0.0, z_cm=300.0,
                              intensity_lumens=4000.0, attenuation_cm=800.0)

            # Camera parked ~2m from origin, looking DOWN-LEFT at dog at (0,0,0)
            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            import math as _m
            cx, cy, cz = 180.0, 0.0, 60.0  # 1.8m in front, 0.6m high
            look_x, look_y, look_z = 0.0, 0.0, 30.0
            yaw = _m.degrees(_m.atan2(look_y - cy, look_x - cx))   # 180
            pitch = -_m.degrees(_m.atan2(cz - look_z, _m.hypot(look_x - cx, look_y - cy)))
            cam.K2_SetActorLocationAndRotation(
                NewLocation={"X": cx, "Y": cy, "Z": cz},
                NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                bSweep=False, bTeleport=True,
            )

            # Spawn dog at origin
            bp = game.unreal_service.load_class(uclass="AActor", name=ANIMATED_DOG_BP)
            actor = game.unreal_service.spawn_actor(
                uclass=bp,
                location={"X": 0.0, "Y": 0.0, "Z": 0.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            actor.SetActorScale3D(
                NewScale3D={"X": args.dog_scale, "Y": args.dog_scale, "Z": args.dog_scale}
            )
            # Face the camera (rotate 90 so the side is toward camera)
            actor.K2_SetActorLocationAndRotation(
                NewLocation={"X": 0.0, "Y": 0.0, "Z": 0.0},
                NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": 90.0},  # side-on
                bSweep=False, bTeleport=True,
            )
            actor.SetActorTickEnabled(bEnabled=True)

            smc = game.unreal_service.get_component_by_class(actor=actor, uclass="USkeletalMeshComponent")
            smc.SetComponentTickEnabled(bEnabled=True)
            anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=WALKING_ANIM)
            smc.PlayAnimation(NewAnimToPlay=anim, bLooping=True)
            try:
                smc.SetPlayRate(Rate=1.0)
            except Exception:
                pass

            # THE ACTUAL ROOT CAUSE — SPEAR starts the world PAUSED
            # (see SpGameMode / SpPauseManager). No SkeletalMeshComponent
            # tick can advance while paused, so anim clock never moves.
            # examples/control_character/run.py calls this exact line
            # (run.py:142) right before its per-frame loop.
            gameplay_statics = game.get_unreal_object(uclass="UGameplayStatics")
            gameplay_statics.SetGamePaused(bPaused=False)
            spear.log("SetGamePaused(False) called")
        with instance.end_frame():
            pass

        # VT + shader warm-up
        instance.step(num_frames=args.warmup)

        # Render frames, WITHOUT touching the dog transform. Any per-frame
        # change must come from anim tick (or from lighting jitter).
        for i in range(args.n_frames):
            instance.step(num_frames=args.per_frame_warmup)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                img = read_frame(comp)
                cv2.imwrite(os.path.join(args.output_dir, f"frame_{i:04d}.png"), img)

        print(f"RENDER_DONE {args.output_dir}", flush=True)
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
