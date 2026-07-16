#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Diagnose whether SceneCapture show-flag updates hide Entry BSP.

This renders the same minimal Entry scene several ways and saves both images
and a simple gray-region mask report to /tmp/diag_scene_capture_showflags.
"""

import os
import sys
import traceback

import cv2
import numpy as np


EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__) + "/../examples")
sys.path.insert(0, EXAMPLES_DIR)

from render_in_apartment import CAMERA_BP, read_frame
from render_in_gpurir_room import (
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_sky,
)


OUT_DIR = "/tmp/diag_scene_capture_showflags"
FLAGS = [
    {"ShowFlagName": "BSP", "Enabled": False},
    {"ShowFlagName": "BSPTriangles", "Enabled": False},
]
CAM_LOC = {"X": 460.0, "Y": 390.0, "Z": 73.4}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.3, "Yaw": 180.0}


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


def spawn_camera_variant(game, mode):
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name=CAMERA_BP)
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(
        actor=cam,
        component_name="DefaultSceneRoot.final_tone_curve_hdr_",
        uclass="USpSceneCaptureComponent2D",
    )
    viewport_desc = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(
        camera_sensor=cam,
        camera_components=[comp],
        viewport_desc=viewport_desc,
        widths=1280,
        heights=720,
    )

    calls = []
    if mode == "property_before":
        comp.set_property_value(property_name="ShowFlagSettings", property_value=FLAGS)
        calls.append("set_property_before")
    if mode == "setter_before":
        comp.SetShowFlagSettings(InShowFlagSettings=FLAGS)
        calls.append("SetShowFlagSettings_before")

    comp.Initialize()
    comp.initialize_sp_funcs()
    calls.append("Initialize")

    if mode == "property_after":
        comp.set_property_value(property_name="ShowFlagSettings", property_value=FLAGS)
        calls.append("set_property_after")
    if mode == "setter_after":
        comp.SetShowFlagSettings(InShowFlagSettings=FLAGS)
        calls.append("SetShowFlagSettings_after")
    if mode == "showonly_empty":
        comp.PrimitiveRenderMode = "PRM_UseShowOnlyList"
        comp.ShowOnlyActors = []
        calls.append("ShowOnlyActors_empty")

    return cam, comp, calls


def gray_region_stats(frame):
    rgb = frame[:, :, :3]
    channel_span = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean = rgb.mean(axis=2)
    mask = (channel_span < 28) & (mean > 35) & (mean < 235)
    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area >= 100:
            components.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                }
            )
    components.sort(key=lambda item: item["area"], reverse=True)
    return int(mask.sum()), components[:5], mask_u8


def run_mode(mode, port):
    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game)
            spawn_sky(game=game)
            spawn_directional_light(
                game=game, yaw_deg=-90.0, pitch_deg=-40.0, intensity_lux=10.0
            )
            spawn_point_light(
                game=game,
                x_cm=260.0,
                y_cm=220.0,
                z_cm=265.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            cam, comp, calls = spawn_camera_variant(game=game, mode=mode)
            if mode == "console_show_bsp":
                game.unreal_service.execute_console_command(command="show BSP")
                calls.append("console:show BSP")
            if mode == "console_r_bsp_0":
                game.unreal_service.execute_console_command(command="r.ShowFlags.BSP 0")
                game.unreal_service.execute_console_command(command="r.ShowFlags.BSPTriangles 0")
                calls.append("console:r.ShowFlags.BSP/BSPTriangles 0")
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

        image_path = os.path.join(OUT_DIR, f"{mode}.png")
        mask_path = os.path.join(OUT_DIR, f"{mode}_gray_mask.png")
        cv2.imwrite(image_path, frame)
        gray_pixels, components, mask = gray_region_stats(frame)
        cv2.imwrite(mask_path, mask)
        print(
            f"MODE {mode} calls={calls} gray_pixels={gray_pixels} components={components}",
            flush=True,
        )
    finally:
        instance.close(force=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = [
        "baseline",
        "property_after",
        "property_before",
        "setter_after",
        "setter_before",
        "console_r_bsp_0",
        "console_show_bsp",
        "showonly_empty",
    ]
    for i, mode in enumerate(modes):
        try:
            run_mode(mode=mode, port=39100 + i)
        except Exception:
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
