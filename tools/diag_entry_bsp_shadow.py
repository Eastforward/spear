#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Diagnose whether hidden Entry BSP still casts shadows into GPURIR renders."""

import os
import sys
import traceback

import cv2
import numpy as np


EXAMPLES_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "examples"))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from render_in_apartment import read_frame, spawn_camera
from render_in_gpurir_room import (
    GLASS_THICKNESS_M,
    M2CM,
    WALL_THICKNESS_M,
    compute_shoebox_room_layout,
    compute_window_frame_layout,
    compute_window_wall_layout,
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_room_piece,
    spawn_sky,
)


OUT_DIR = "/tmp/diag_entry_bsp_shadow"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_W_M = 1.4
WINDOW_H_M = 1.4
WINDOW_Z_BOTTOM_M = 0.9
WINDOW_FRAME_THICKNESS_M = 0.06
CAM_LOC = {"X": 456.9615506024416, "Y": 355.27036446661396, "Z": 73.4}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.309932474020213, "Yaw": 170.0}
SHADOW_ROI = (980, 165, 1265, 410)
REFERENCE_ROI = (650, 165, 935, 410)


def material_for_piece(name):
    if name == "floor":
        return "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"
    elif name == "window_glass":
        return (
            "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/"
            "MI_Casement_Glass.MI_Casement_Glass"
        )
    elif name.startswith("window_frame_"):
        return "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Casement.MI_Casement"
    elif name == "outdoor_ground":
        return "/Game/StarterContent/Materials/M_Ground_Grass.M_Ground_Grass"
    else:
        return "/Game/StarterContent/Materials/M_Basic_Wall.M_Basic_Wall"


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


def entry_model_components(game):
    world = game.get_unreal_object(uobject=game.get_world(), with_sp_funcs=False)
    raw_level = world.get_property_value("PersistentLevel", as_raw_value=True)
    level = game.get_unreal_object(uobject=int(raw_level.value, 16), with_sp_funcs=False)
    raw_components = level.get_property_value("ModelComponents", as_raw_value=True)
    return [
        game.get_unreal_object(uobject=int(handle, 16), with_sp_funcs=False)
        for handle in raw_components.value
    ]


def set_entry_model_components_cast_shadow(game, enabled):
    components = entry_model_components(game=game)
    for comp in components:
        comp.SetCastShadow(NewCastShadow=bool(enabled))
        comp.SetCastHiddenShadow(NewCastHiddenShadow=False)
        comp.bCastDynamicShadow = bool(enabled)
        comp.bCastStaticShadow = bool(enabled)
    return len(components)


def shadow_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    x0, y0, x1, y1 = SHADOW_ROI
    sx0, sy0, sx1, sy1 = REFERENCE_ROI
    shadow_mean = float(gray[y0:y1,x0:x1].mean())
    reference_mean = float(gray[sy0:sy1,sx0:sx1].mean())
    return reference_mean - shadow_mean, reference_mean, shadow_mean


def write_roi_image(frame, path):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = SHADOW_ROI
    sx0, sy0, sx1, sy1 = REFERENCE_ROI
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.rectangle(bgr, (sx0, sy0), (sx1, sy1), (0, 255, 0), 2)
    cv2.imwrite(path, bgr)


def run_mode(mode, port):
    instance = configure_gpurir_instance(rpc_port=port)
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
            point_light = spawn_point_light(
                game=game,
                x_cm=ROOM_SIZE_M[0]*M2CM/2.0,
                y_cm=ROOM_SIZE_M[1]*M2CM/2.0,
                z_cm=ROOM_SIZE_M[2]*M2CM - 15.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            if mode == "point_no_shadow":
                light_comp = game.unreal_service.get_component_by_class(
                    actor=point_light,
                    uclass="UPointLightComponent",
                )
                light_comp.SetCastShadows(bNewValue=False)
            model_component_count = 0
            if mode == "model_no_shadow":
                model_component_count = set_entry_model_components_cast_shadow(
                    game=game,
                    enabled=False,
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

        image_path = os.path.join(OUT_DIR, f"{mode}.png")
        roi_path = os.path.join(OUT_DIR, f"{mode}_roi.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        write_roi_image(frame=frame, path=roi_path)
        score, reference_mean, shadow_mean = shadow_score(frame=frame)
        print(
            "MODE "
            f"{mode} score={score:.3f} reference_mean={reference_mean:.3f} "
            f"shadow_mean={shadow_mean:.3f} model_components={model_component_count} "
            f"image={image_path} roi={roi_path}",
            flush=True,
        )
    finally:
        instance.close(force=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = ["current", "point_no_shadow", "model_no_shadow"]
    for i, mode in enumerate(modes):
        try:
            run_mode(mode=mode, port=39230 + i)
        except Exception:
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
