#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Verify that Entry map BSP does not leak into GPURIR captures."""

import os
import sys

import cv2
import numpy as np


EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__) + "/../examples")
sys.path.insert(0, EXAMPLES_DIR)

from render_in_apartment import read_frame, spawn_camera
from render_in_gpurir_room import (
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_sky,
)


OUT_DIR = "/tmp/verify_entry_bsp_hidden"
CAM_LOC = {"X": 460.0, "Y": 390.0, "Z": 73.4}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.3, "Yaw": 180.0}
MAX_GRAY_COMPONENT_AREA = 1000


def destroy_entry_noise(game):
    for cls in (
        "APlayerStart",
        "ADefaultPawn",
        "ASpectatorPawn",
        "AStaticMeshActor",
        "ASkeletalMeshActor",
        "ABrush",
        "ADecalActor",
        "AInstancedFoliageActor",
        "AGameplayDebuggerCategoryReplicator",
        "AGameplayDebuggerPlayerManager",
        "ANavigationData",
        "ANavMeshBoundsVolume",
    ):
        try:
            victims = game.unreal_service.find_actors_by_class(uclass=cls)
        except Exception:
            victims = []
        for actor in victims:
            try:
                game.unreal_service.destroy_actor(actor=actor)
            except Exception:
                pass


def largest_gray_component_area(frame):
    rgb = frame[:,:,:3]
    channel_span = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean = rgb.mean(axis=2)
    mask = ((channel_span < 28) & (mean > 35) & (mean < 235)).astype(np.uint8)*255
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return 0, mask
    else:
        return int(stats[1:,cv2.CC_STAT_AREA].max()), mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    instance = configure_gpurir_instance(rpc_port=39180)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            spawn_sky(game=game)
            spawn_directional_light(
                game=game,
                yaw_deg=-90.0,
                pitch_deg=-40.0,
                intensity_lux=10.0,
            )
            spawn_point_light(
                game=game,
                x_cm=260.0,
                y_cm=220.0,
                z_cm=265.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame():
            pass

        instance.step(num_frames=4)
        with instance.begin_frame():
            cam.K2_SetActorLocationAndRotation(
                NewLocation=CAM_LOC,
                NewRotation=CAM_ROT,
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=25)
        with instance.begin_frame():
            pass
        with instance.end_frame():
            frame = read_frame(comp)

        image_path = os.path.join(OUT_DIR, "frame.png")
        mask_path = os.path.join(OUT_DIR, "gray_mask.png")
        cv2.imwrite(image_path, frame)
        max_area, mask = largest_gray_component_area(frame=frame)
        cv2.imwrite(mask_path, mask)
        print(f"VERIFY_ENTRY_BSP max_gray_component_area={max_area}", flush=True)
        print(f"VERIFY_ENTRY_BSP image={image_path}", flush=True)
        print(f"VERIFY_ENTRY_BSP mask={mask_path}", flush=True)
        assert max_area < MAX_GRAY_COMPONENT_AREA
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
