#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Top-down diagnosis for the left/back floor dark area in GPURIR room renders."""

import os
import sys
import traceback

import cv2


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


OUT_DIR = "/tmp/diag_user_floor_shadow_topdown"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_W_M = 1.4
WINDOW_H_M = 1.4
WINDOW_Z_BOTTOM_M = 0.9
WINDOW_FRAME_THICKNESS_M = 0.06


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
    window_cx_m = ROOM_SIZE_M[0] / 2.0
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
    non_y1 = [p for p in room_pieces if p["name"] != "wall_y1"]
    ry_cm = ROOM_SIZE_M[1] * M2CM
    t_cm = WALL_THICKNESS_M * M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx_m * M2CM,
            ry_cm + t_cm / 2.0,
            (WINDOW_Z_BOTTOM_M + WINDOW_H_M / 2.0) * M2CM,
        ),
        "scale": (WINDOW_W_M, GLASS_THICKNESS_M, WINDOW_H_M),
    }
    return non_y1 + window_pieces + window_frame_pieces + [glass_piece]


def set_actor_cast_shadow(game, actor, enabled):
    comp = game.unreal_service.get_component_by_class(
        actor=actor,
        uclass="UStaticMeshComponent",
    )
    comp.SetCastShadow(NewCastShadow=bool(enabled))
    try:
        comp.SetCastHiddenShadow(NewCastHiddenShadow=False)
    except Exception:
        pass
    try:
        comp.bCastDynamicShadow = bool(enabled)
    except Exception:
        pass
    try:
        comp.bCastStaticShadow = bool(enabled)
    except Exception:
        pass


def set_light_cast_shadow(game, actor, uclass, enabled):
    comp = game.unreal_service.get_component_by_class(actor=actor, uclass=uclass)
    comp.SetCastShadows(bNewValue=bool(enabled))


def spawn_room(game):
    actors = {}
    for piece in gpurir_room_pieces():
        actor = spawn_room_piece(
            game=game,
            piece=piece,
            material_path=_material_for_piece(piece["name"]),
        )
        actors[piece["name"]] = actor
    return actors


def apply_mode(game, mode, room_actors, dir_light):
    changed = []
    if mode == "ceiling_no_shadow":
        names = ("ceiling",)
    elif mode == "wall_y1_top_no_shadow":
        names = ("wall_y1_top",)
    elif mode == "wall_y1_right_no_shadow":
        names = ("wall_y1_right",)
    elif mode == "wall_y1_left_no_shadow":
        names = ("wall_y1_left",)
    elif mode == "wall_y1_bottom_no_shadow":
        names = ("wall_y1_bottom",)
    elif mode == "wall_y1_top_right_no_shadow":
        names = ("wall_y1_top", "wall_y1_right")
    elif mode == "window_frame_no_shadow":
        names = tuple(
            name for name in room_actors
            if name.startswith("window_frame_") or name == "window_glass"
        )
    elif mode == "wall_y1_all_no_shadow":
        names = tuple(
            name for name in room_actors
            if name.startswith("wall_y1_")
        )
    elif mode == "window_wall_no_shadow":
        names = tuple(
            name for name in room_actors
            if (
                name.startswith("wall_y1_")
                or name.startswith("window_frame_")
                or name == "window_glass"
            )
        )
    else:
        names = ()
    if names:
        for name, actor in room_actors.items():
            if name in names:
                set_actor_cast_shadow(game=game, actor=actor, enabled=False)
                changed.append(name)
    elif mode == "directional_no_shadow":
        set_light_cast_shadow(
            game=game,
            actor=dir_light,
            uclass="UDirectionalLightComponent",
            enabled=False,
        )
        changed.append("UDirectionalLightComponent")
    return changed


def render_mode(mode, port):
    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            room_actors = spawn_room(game=game)
            spawn_sky(game=game)
            dir_light = spawn_directional_light(
                game=game,
                yaw_deg=-90.0,
                pitch_deg=-40.0,
                intensity_lux=10.0,
            )
            spawn_point_light(
                game=game,
                x_cm=ROOM_SIZE_M[0] * M2CM / 2.0,
                y_cm=ROOM_SIZE_M[1] * M2CM / 2.0,
                z_cm=ROOM_SIZE_M[2] * M2CM - 15.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            changed = apply_mode(
                game=game,
                mode=mode,
                room_actors=room_actors,
                dir_light=dir_light,
            )
            cam, comp = spawn_camera(game=game, width=1280, height=720)
            try:
                comp.ProjectionType = "Orthographic"
                comp.OrthoWidth = 850.0
            except Exception as e:
                print(f"MODE {mode} warn orthographic failed: {e}", flush=True)
            cam.K2_SetActorLocationAndRotation(
                NewLocation={"X": 260.0, "Y": 220.0, "Z": 250.0},
                NewRotation={"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0},
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
        print(
            f"MODE {mode} changed={changed} image={image_path}",
            flush=True,
        )
    finally:
        instance.close(force=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = [
        "baseline",
        "ceiling_no_shadow",
        "wall_y1_top_no_shadow",
        "wall_y1_right_no_shadow",
        "wall_y1_left_no_shadow",
        "wall_y1_bottom_no_shadow",
        "wall_y1_top_right_no_shadow",
        "window_frame_no_shadow",
        "wall_y1_all_no_shadow",
        "window_wall_no_shadow",
        "directional_no_shadow",
    ]
    for i, mode in enumerate(modes):
        try:
            render_mode(mode=mode, port=39410 + i)
        except Exception:
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
