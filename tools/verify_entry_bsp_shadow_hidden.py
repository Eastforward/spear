#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Verify that hidden Entry BSP does not cast a residual GPURIR-room shadow."""

import os
import sys

import cv2


TOOLS_DIR = os.path.realpath(os.path.dirname(__file__))
EXAMPLES_DIR = os.path.realpath(os.path.join(TOOLS_DIR, "..", "examples"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from diag_entry_bsp_shadow import (
    CAM_LOC,
    CAM_ROT,
    M2CM,
    ROOM_SIZE_M,
    destroy_entry_noise,
    gpurir_room_pieces,
    material_for_piece,
    shadow_score,
    write_roi_image,
)
from render_in_apartment import read_frame, spawn_camera
from render_in_gpurir_room import (
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_room_piece,
    spawn_sky,
)


OUT_DIR = "/tmp/verify_entry_bsp_shadow_hidden"
MAX_SHADOW_SCORE = 26.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    instance = configure_gpurir_instance(rpc_port=39260)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            for piece in gpurir_room_pieces():
                spawn_room_piece(
                    game=game,
                    piece=piece,
                    material_path=material_for_piece(name=piece["name"]),
                    cast_shadow=True,
                )
            spawn_sky(game=game)
            spawn_directional_light(
                game=game,
                yaw_deg=-90.0,
                pitch_deg=-40.0,
                intensity_lux=10.0,
            )
            spawn_point_light(
                game=game,
                x_cm=ROOM_SIZE_M[0]*M2CM/2.0,
                y_cm=ROOM_SIZE_M[1]*M2CM/2.0,
                z_cm=ROOM_SIZE_M[2]*M2CM - 15.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame():
            pass

        instance.step(num_frames=10)
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
        roi_path = os.path.join(OUT_DIR, "roi.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        write_roi_image(frame=frame, path=roi_path)
        score, reference_mean, shadow_mean = shadow_score(frame=frame)
        print(f"VERIFY_ENTRY_BSP_SHADOW score={score:.3f}", flush=True)
        print(f"VERIFY_ENTRY_BSP_SHADOW reference_mean={reference_mean:.3f}", flush=True)
        print(f"VERIFY_ENTRY_BSP_SHADOW shadow_mean={shadow_mean:.3f}", flush=True)
        print(f"VERIFY_ENTRY_BSP_SHADOW image={image_path}", flush=True)
        print(f"VERIFY_ENTRY_BSP_SHADOW roi={roi_path}", flush=True)
        assert score <= MAX_SHADOW_SCORE
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
