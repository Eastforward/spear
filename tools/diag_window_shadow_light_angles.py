#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Diagnose GPURIR floor shadow sensitivity to directional-light angle."""

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


OUT_DIR = "/tmp/diag_window_shadow_light_angles"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_W_M = 1.4
WINDOW_H_M = 1.4
WINDOW_Z_BOTTOM_M = 0.9
WINDOW_FRAME_THICKNESS_M = 0.06
CAM_LOC = {"X": 63.03844939755839, "Y": 424.7296355333861, "Z": 73.4}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.309932474020213, "Yaw": -10.0}
ROIS = {
    "upper_dark": (20, 320, 610, 400),
    "sun_patch": (160, 405, 560, 520),
    "frame_shadow_line": (150, 345, 600, 370),
    "lit_near_line": (190, 400, 550, 430),
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


def render_mode(mode, yaw_deg, pitch_deg, port):
    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            for piece in gpurir_room_pieces():
                spawn_room_piece(
                    game=game,
                    piece=piece,
                    material_path=_material_for_piece(piece["name"]),
                    cast_shadow=True,
                )
            spawn_sky(game=game)
            spawn_directional_light(
                game=game,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
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
        print(f"MODE {mode} yaw={yaw_deg} pitch={pitch_deg} {' '.join(values)} image={image_path}", flush=True)
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
        ("pitch_m10", -90.0, -10.0),
        ("pitch_m20", -90.0, -20.0),
        ("pitch_m30", -90.0, -30.0),
        ("pitch_m40", -90.0, -40.0),
        ("yaw_m75_pitch_m30", -75.0, -30.0),
        ("yaw_m105_pitch_m30", -105.0, -30.0),
    ]
    frames = {}
    for i, (mode, yaw_deg, pitch_deg) in enumerate(modes):
        try:
            frames[mode] = render_mode(
                mode=mode,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                port=39570 + i,
            )
        except Exception:
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    if frames:
        write_contact_sheet(frames=frames, modes=[ mode for mode, _yaw, _pitch in modes if mode in frames ])
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
