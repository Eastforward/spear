#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Diagnose which GPURIR room pieces cast the user-reported floor shadow."""

import os
import sys
import traceback

import cv2
import numpy as np


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from render_in_apartment import read_frame, spawn_camera
from render_in_gpurir_room import (
    GLASS_THICKNESS_M,
    M2CM,
    WALL_THICKNESS_M,
    _material_for_piece,
    compute_shoebox_room_layout,
    compute_window_frame_layout,
    compute_window_wall_layout,
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_room_piece,
    spawn_sky,
)


OUT_DIR = "/tmp/diag_window_shadow_pieces"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_W_M = 1.4
WINDOW_H_M = 1.4
WINDOW_Z_BOTTOM_M = 0.9
WINDOW_FRAME_THICKNESS_M = 0.06
CAM_LOC = {"X": 63.03844939755839, "Y": 424.7296355333861, "Z": 73.4}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.309932474020213, "Yaw": -10.0}
ROIS = {
    "user_box": (20, 330, 650, 540),
    "upper_dark": (20, 320, 610, 400),
    "sun_patch": (160, 405, 560, 520),
}


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


def gpurir_room_pieces():
    room_pieces = compute_shoebox_room_layout(
        room_size_m=ROOM_SIZE_M,
        wall_thickness_m=WALL_THICKNESS_M,
    )
    window_cx_m = ROOM_SIZE_M[0]/2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=ROOM_SIZE_M,
        window_w_m=WINDOW_W_M,
        window_h_m=WINDOW_H_M,
        window_cx_m=window_cx_m,
        window_z_bottom_m=WINDOW_Z_BOTTOM_M,
        wall_thickness_m=WALL_THICKNESS_M,
    )
    window_frame_pieces = compute_window_frame_layout(
        room_size_m=ROOM_SIZE_M,
        window_w_m=WINDOW_W_M,
        window_h_m=WINDOW_H_M,
        window_cx_m=window_cx_m,
        window_z_bottom_m=WINDOW_Z_BOTTOM_M,
        wall_thickness_m=WALL_THICKNESS_M,
        frame_thickness_m=WINDOW_FRAME_THICKNESS_M,
    )
    non_y1 = [ p for p in room_pieces if p["name"] != "wall_y1" ]
    ry_cm = ROOM_SIZE_M[1]*M2CM
    t_cm = WALL_THICKNESS_M*M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx_m*M2CM,
            ry_cm + t_cm/2.0,
            (WINDOW_Z_BOTTOM_M + WINDOW_H_M/2.0)*M2CM,
        ),
        "scale": (WINDOW_W_M, GLASS_THICKNESS_M, WINDOW_H_M),
    }
    outdoor_ground_piece = {
        "name": "outdoor_ground",
        "location_cm": (
            ROOM_SIZE_M[0]*M2CM/2.0,
            ROOM_SIZE_M[1]*M2CM/2.0,
            -WALL_THICKNESS_M*M2CM - 5.0,
        ),
        "scale": (80.0, 80.0, 0.1),
    }
    return non_y1 + window_pieces + window_frame_pieces + [glass_piece, outdoor_ground_piece]


def disabled_piece_names(mode):
    if mode == "baseline":
        return ()
    elif mode == "no_wall_y1_top":
        return ("wall_y1_top",)
    elif mode == "no_wall_y1_right":
        return ("wall_y1_right",)
    elif mode == "no_wall_y1_left":
        return ("wall_y1_left",)
    elif mode == "no_wall_y1_bottom":
        return ("wall_y1_bottom",)
    elif mode == "no_wall_y1_top_right":
        return ("wall_y1_top", "wall_y1_right")
    elif mode == "no_wall_y1_top_left":
        return ("wall_y1_top", "wall_y1_left")
    elif mode == "no_wall_y1_top_bottom":
        return ("wall_y1_top", "wall_y1_bottom")
    elif mode == "no_wall_y1_top_left_bottom":
        return ("wall_y1_top", "wall_y1_left", "wall_y1_bottom")
    elif mode == "no_wall_x0":
        return ("wall_x0",)
    elif mode == "no_wall_x1":
        return ("wall_x1",)
    elif mode == "no_wall_y0":
        return ("wall_y0",)
    elif mode == "no_wall_x0_y0":
        return ("wall_x0", "wall_y0")
    elif mode == "no_window_frame":
        return (
            "window_frame_top",
            "window_frame_bottom",
            "window_frame_left",
            "window_frame_right",
            "window_glass",
        )
    elif mode == "no_ceiling":
        return ("ceiling",)
    elif mode == "no_wall_y1_all":
        return ("wall_y1_bottom", "wall_y1_top", "wall_y1_left", "wall_y1_right")
    else:
        return ()


def render_mode(mode, port):
    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        disabled = disabled_piece_names(mode=mode)
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            for piece in gpurir_room_pieces():
                spawn_room_piece(
                    game=game,
                    piece=piece,
                    material_path=_material_for_piece(piece["name"]),
                    cast_shadow=piece["name"] not in disabled,
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
            cam.K2_SetActorLocationAndRotation(
                NewLocation=CAM_LOC,
                NewRotation=CAM_ROT,
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=30)
        with instance.begin_frame():
            pass
        with instance.end_frame():
            frame = read_frame(comp)

        image_path = os.path.join(OUT_DIR, f"{mode}.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        values = []
        for name, (x0, y0, x1, y1) in ROIS.items():
            values.append(f"{name}={gray[y0:y1,x0:x1].mean():.3f}")
        print(f"MODE {mode} disabled={disabled} {' '.join(values)} image={image_path}", flush=True)
        return frame
    finally:
        instance.close(force=True)


def write_contact_sheet(frames, modes):
    thumbs = []
    for mode in modes:
        frame = frames[mode]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for name, (x0, y0, x1, y1) in ROIS.items():
            color = (0, 0, 255) if name != "sun_patch" else (0, 255, 0)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        thumb = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, mode, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    rows = []
    for i in range(0, len(thumbs), 3):
        row = thumbs[i:i + 3]
        while len(row) < 3:
            row.append(np.full_like(thumbs[0], 255))
        rows.append(np.hstack(row))
    cv2.imwrite(os.path.join(OUT_DIR, "contact.png"), np.vstack(rows))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = [
        "baseline",
        "no_wall_y1_top",
        "no_wall_y1_right",
        "no_wall_y1_left",
        "no_wall_y1_bottom",
        "no_wall_y1_top_right",
        "no_wall_y1_top_left",
        "no_wall_y1_top_bottom",
        "no_wall_y1_top_left_bottom",
        "no_wall_x0",
        "no_wall_x1",
        "no_wall_y0",
        "no_wall_x0_y0",
        "no_window_frame",
        "no_ceiling",
        "no_wall_y1_all",
    ]
    frames = {}
    for i, mode in enumerate(modes):
        try:
            frames[mode] = render_mode(mode=mode, port=39520 + i)
        except Exception:
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    if frames:
        write_contact_sheet(frames=frames, modes=[ mode for mode in modes if mode in frames ])
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
