#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Compare physical GPURIR tuning options without disabling wall shadows."""

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
    WALL_MATERIAL,
    WALL_THICKNESS_M,
    _material_for_piece,
    compute_shoebox_room_layout,
    compute_window_frame_layout,
    compute_window_wall_layout,
    configure_gpurir_instance,
    piece_casts_shadow,
    spawn_directional_light,
    spawn_point_light,
    spawn_room_piece,
    spawn_sky,
)


OUT_DIR = "/tmp/diag_gpurir_tuning_matrix"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_FRAME_THICKNESS_M = 0.06
IMAGE_W = 1280
IMAGE_H = 720
ORTHO_WIDTH_CM = 850.0
TOPDOWN_LOC = {"X": 260.0, "Y": 220.0, "Z": 250.0}
TOPDOWN_ROT = {"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0}
PERSPECTIVE_LOC = {"X": 63.03844939755839, "Y": 424.7296355333861, "Z": 73.4}
PERSPECTIVE_ROT = {"Roll": 0.0, "Pitch": -11.309932474020213, "Yaw": -10.0}
FLOOR_ARTIFACT_ROI = (450, 60, 620, 310)
SUN_PATCH_ROI = (570, 260, 800, 470)
PERSPECTIVE_USER_ROI = (20, 320, 610, 400)
PERSPECTIVE_SUN_ROI = (160, 405, 560, 520)


MODES = [
    {
        "name": "baseline",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "plain_floor_debug",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 2200.0,
        "floor_material": WALL_MATERIAL,
    },
    {
        "name": "ceiling_light_6000",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 6000.0,
        "floor_material": None,
    },
    {
        "name": "ceiling_light_10000",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 10000.0,
        "floor_material": None,
    },
    {
        "name": "sun_pitch_m35",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -35.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "sun_pitch_m30",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -30.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "sun_yaw_m100_pitch_m35",
        "window_w_m": 1.4,
        "window_h_m": 1.4,
        "window_z_bottom_m": 0.9,
        "light_yaw": -100.0,
        "light_pitch": -35.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "window_h_155",
        "window_w_m": 1.4,
        "window_h_m": 1.55,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "window_h_170",
        "window_w_m": 1.4,
        "window_h_m": 1.7,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -40.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
    {
        "name": "window_h_155_pitch_m35",
        "window_w_m": 1.4,
        "window_h_m": 1.55,
        "window_z_bottom_m": 0.9,
        "light_yaw": -90.0,
        "light_pitch": -35.0,
        "ceiling_lumens": 2200.0,
        "floor_material": None,
    },
]


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


def room_pieces(mode):
    shoebox_pieces = compute_shoebox_room_layout(
        room_size_m=ROOM_SIZE_M,
        wall_thickness_m=WALL_THICKNESS_M,
    )
    window_cx_m = ROOM_SIZE_M[0] / 2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=ROOM_SIZE_M,
        window_w_m=mode["window_w_m"],
        window_h_m=mode["window_h_m"],
        window_cx_m=window_cx_m,
        window_z_bottom_m=mode["window_z_bottom_m"],
        wall_thickness_m=WALL_THICKNESS_M,
    )
    window_frame_pieces = compute_window_frame_layout(
        room_size_m=ROOM_SIZE_M,
        window_w_m=mode["window_w_m"],
        window_h_m=mode["window_h_m"],
        window_cx_m=window_cx_m,
        window_z_bottom_m=mode["window_z_bottom_m"],
        wall_thickness_m=WALL_THICKNESS_M,
        frame_thickness_m=WINDOW_FRAME_THICKNESS_M,
    )
    non_y1 = [p for p in shoebox_pieces if p["name"] != "wall_y1"]
    ry_cm = ROOM_SIZE_M[1] * M2CM
    t_cm = WALL_THICKNESS_M * M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx_m * M2CM,
            ry_cm + t_cm / 2.0,
            (mode["window_z_bottom_m"] + mode["window_h_m"] / 2.0) * M2CM,
        ),
        "scale": (mode["window_w_m"], GLASS_THICKNESS_M, mode["window_h_m"]),
    }
    return non_y1 + window_pieces + window_frame_pieces + [glass_piece]


def roi_mean(frame, roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    x0, y0, x1, y1 = roi
    return float(gray[y0:y1, x0:x1].mean())


def draw_rois(frame, mode_name, *, topdown):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rois = (
        (
            (FLOOR_ARTIFACT_ROI, (0, 0, 255), "floor_artifact"),
            (SUN_PATCH_ROI, (0, 255, 0), "sun_patch"),
        )
        if topdown
        else (
            (PERSPECTIVE_USER_ROI, (0, 0, 255), "user_box"),
            (PERSPECTIVE_SUN_ROI, (0, 255, 0), "sun_patch"),
        )
    )
    for roi, color, label in rois:
        x0, y0, x1, y1 = roi
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            bgr,
            label,
            (x0, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        mode_name,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def spawn_scene(game, mode):
    destroy_entry_noise(game=game)
    for piece in room_pieces(mode=mode):
        material_path = _material_for_piece(piece["name"])
        if piece["name"] == "floor" and mode["floor_material"]:
            material_path = mode["floor_material"]
        spawn_room_piece(
            game=game,
            piece=piece,
            material_path=material_path,
            cast_shadow=piece_casts_shadow(
                name=piece["name"],
                ceiling_casts_shadow=True,
                window_top_wall_casts_shadow=True,
                window_wall_casts_shadow=True,
            ),
        )
    spawn_sky(game=game)
    spawn_directional_light(
        game=game,
        yaw_deg=mode["light_yaw"],
        pitch_deg=mode["light_pitch"],
        intensity_lux=10.0,
    )
    spawn_point_light(
        game=game,
        x_cm=ROOM_SIZE_M[0] * M2CM / 2.0,
        y_cm=ROOM_SIZE_M[1] * M2CM / 2.0,
        z_cm=ROOM_SIZE_M[2] * M2CM - 15.0,
        intensity_lumens=mode["ceiling_lumens"],
        attenuation_cm=600.0,
    )
    top_cam, top_comp = spawn_camera(game=game, width=IMAGE_W, height=IMAGE_H)
    try:
        top_comp.ProjectionType = "Orthographic"
        top_comp.OrthoWidth = ORTHO_WIDTH_CM
    except Exception as e:
        print(f"MODE {mode['name']} warn orthographic failed: {e}", flush=True)
    top_cam.K2_SetActorLocationAndRotation(
        NewLocation=TOPDOWN_LOC,
        NewRotation=TOPDOWN_ROT,
        bSweep=False,
        bTeleport=True,
    )
    persp_cam, persp_comp = spawn_camera(game=game, width=IMAGE_W, height=IMAGE_H)
    persp_cam.K2_SetActorLocationAndRotation(
        NewLocation=PERSPECTIVE_LOC,
        NewRotation=PERSPECTIVE_ROT,
        bSweep=False,
        bTeleport=True,
    )
    return top_comp, persp_comp


def render_mode(mode, port):
    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            top_comp, persp_comp = spawn_scene(game=game, mode=mode)
        with instance.end_frame():
            pass

        instance.step(num_frames=30)
        with instance.begin_frame():
            pass
        with instance.end_frame():
            top_frame = read_frame(top_comp)
            persp_frame = read_frame(persp_comp)

        name = mode["name"]
        top_path = os.path.join(OUT_DIR, f"{name}_topdown.png")
        persp_path = os.path.join(OUT_DIR, f"{name}_perspective.png")
        top_roi_path = os.path.join(OUT_DIR, f"{name}_topdown_roi.png")
        persp_roi_path = os.path.join(OUT_DIR, f"{name}_perspective_roi.png")
        cv2.imwrite(top_path, cv2.cvtColor(top_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(persp_path, cv2.cvtColor(persp_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(top_roi_path, draw_rois(top_frame, name, topdown=True))
        cv2.imwrite(persp_roi_path, draw_rois(persp_frame, name, topdown=False))

        floor_mean = roi_mean(top_frame, FLOOR_ARTIFACT_ROI)
        sun_mean = roi_mean(top_frame, SUN_PATCH_ROI)
        persp_user_mean = roi_mean(persp_frame, PERSPECTIVE_USER_ROI)
        persp_sun_mean = roi_mean(persp_frame, PERSPECTIVE_SUN_ROI)
        print(
            f"MODE {name} "
            f"floor_artifact_mean={floor_mean:.3f} "
            f"sun_patch_mean={sun_mean:.3f} "
            f"gap={sun_mean - floor_mean:.3f} "
            f"persp_user_mean={persp_user_mean:.3f} "
            f"persp_sun_mean={persp_sun_mean:.3f} "
            f"top={top_roi_path} persp={persp_roi_path}",
            flush=True,
        )
        return cv2.imread(top_roi_path), cv2.imread(persp_roi_path)
    finally:
        instance.close(force=True)


def write_contact(images, filename):
    thumbs = []
    for image, mode in images:
        if image is None:
            image = np.full((IMAGE_H, IMAGE_W, 3), 255, dtype=np.uint8)
            cv2.putText(image, mode, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        thumbs.append(cv2.resize(image, (426, 240), interpolation=cv2.INTER_AREA))
    rows = []
    for i in range(0, len(thumbs), 2):
        row = thumbs[i:i + 2]
        while len(row) < 2:
            row.append(np.full_like(thumbs[0], 255))
        rows.append(np.hstack(row))
    path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(path, np.vstack(rows))
    print(f"CONTACT {path}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    top_images = []
    persp_images = []
    for i, mode in enumerate(MODES):
        try:
            top_image, persp_image = render_mode(mode=mode, port=39920 + i)
        except Exception:
            top_image, persp_image = None, None
            print(f"MODE {mode['name']} FAILED", flush=True)
            traceback.print_exc()
        top_images.append((top_image, mode["name"]))
        persp_images.append((persp_image, mode["name"]))
    write_contact(top_images, "contact_topdown.png")
    write_contact(persp_images, "contact_perspective.png")
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
