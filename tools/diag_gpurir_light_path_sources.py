#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Diagnose GPURIR floor dark/light bands by material swap and light-path tracing."""

import math
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


OUT_DIR = "/tmp/diag_gpurir_light_path_sources"
ROOM_SIZE_M = (5.2, 4.4, 2.8)
WINDOW_W_M = 1.4
WINDOW_H_M = 1.4
WINDOW_Z_BOTTOM_M = 0.9
WINDOW_FRAME_THICKNESS_M = 0.06
ORTHO_WIDTH_CM = 850.0
IMAGE_W = 1280
IMAGE_H = 720
CAM_LOC = {"X": 260.0, "Y": 220.0, "Z": 250.0}
CAM_ROT = {"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0}
FLOOR_ARTIFACT_ROI = (450, 60, 620, 310)
SUN_PATCH_ROI = (570, 260, 800, 470)
LIGHT_YAW_DEG = -90.0
LIGHT_PITCH_DEG = -40.0


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


def room_pieces():
    shoebox_pieces = compute_shoebox_room_layout(
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
    non_y1 = [p for p in shoebox_pieces if p["name"] != "wall_y1"]
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


def set_light_cast_shadows(game, actor, enabled):
    comp = game.unreal_service.get_component_by_class(
        actor=actor,
        uclass="UDirectionalLightComponent",
    )
    comp.SetCastShadows(bNewValue=bool(enabled))


def draw_rois(frame, label):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for roi, color, name in (
        (FLOOR_ARTIFACT_ROI, (0, 0, 255), "floor_artifact"),
        (SUN_PATCH_ROI, (0, 255, 0), "sun_patch"),
    ):
        x0, y0, x1, y1 = roi
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            bgr,
            name,
            (x0, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def roi_mean(frame, roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    x0, y0, x1, y1 = roi
    return float(gray[y0:y1, x0:x1].mean())


def render_mode(mode, port):
    mode_settings = {
        "physical_floor_texture": {
            "floor_material": None,
            "dir_shadows": True,
            "window_top_wall_casts_shadow": True,
        },
        "physical_plain_floor": {
            "floor_material": WALL_MATERIAL,
            "dir_shadows": True,
            "window_top_wall_casts_shadow": True,
        },
        "directional_no_shadow": {
            "floor_material": None,
            "dir_shadows": False,
            "window_top_wall_casts_shadow": True,
        },
        "top_wall_no_shadow": {
            "floor_material": None,
            "dir_shadows": True,
            "window_top_wall_casts_shadow": False,
        },
    }[mode]

    instance = configure_gpurir_instance(rpc_port=port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            destroy_entry_noise(game=game)
            for piece in room_pieces():
                material_path = _material_for_piece(piece["name"])
                if piece["name"] == "floor" and mode_settings["floor_material"]:
                    material_path = mode_settings["floor_material"]
                spawn_room_piece(
                    game=game,
                    piece=piece,
                    material_path=material_path,
                    cast_shadow=piece_casts_shadow(
                        name=piece["name"],
                        ceiling_casts_shadow=True,
                        window_top_wall_casts_shadow=mode_settings[
                            "window_top_wall_casts_shadow"
                        ],
                        window_wall_casts_shadow=True,
                    ),
                )
            spawn_sky(game=game)
            dir_light = spawn_directional_light(
                game=game,
                yaw_deg=LIGHT_YAW_DEG,
                pitch_deg=LIGHT_PITCH_DEG,
                intensity_lux=10.0,
            )
            set_light_cast_shadows(
                game=game,
                actor=dir_light,
                enabled=mode_settings["dir_shadows"],
            )
            spawn_point_light(
                game=game,
                x_cm=ROOM_SIZE_M[0] * M2CM / 2.0,
                y_cm=ROOM_SIZE_M[1] * M2CM / 2.0,
                z_cm=ROOM_SIZE_M[2] * M2CM - 15.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            cam, comp = spawn_camera(game=game, width=IMAGE_W, height=IMAGE_H)
            try:
                comp.ProjectionType = "Orthographic"
                comp.OrthoWidth = ORTHO_WIDTH_CM
            except Exception as e:
                print(f"MODE {mode} warn orthographic failed: {e}", flush=True)
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
        roi_path = os.path.join(OUT_DIR, f"{mode}_roi.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(roi_path, draw_rois(frame=frame, label=mode))
        print(
            f"MODE {mode} "
            f"floor_artifact_mean={roi_mean(frame, FLOOR_ARTIFACT_ROI):.3f} "
            f"sun_patch_mean={roi_mean(frame, SUN_PATCH_ROI):.3f} "
            f"image={image_path} roi={roi_path}",
            flush=True,
        )
        return cv2.imread(roi_path)
    finally:
        instance.close(force=True)


def pixel_to_floor_world(pixel_x, pixel_y):
    cm_per_pixel_x = ORTHO_WIDTH_CM / float(IMAGE_W)
    cm_per_pixel_y = (ORTHO_WIDTH_CM * IMAGE_H / float(IMAGE_W)) / float(IMAGE_H)
    world_y = CAM_LOC["Y"] + (float(pixel_x) - IMAGE_W / 2.0) * cm_per_pixel_x
    world_x = CAM_LOC["X"] + (float(pixel_y) - IMAGE_H / 2.0) * cm_per_pixel_y
    return (world_x, world_y, 0.5)


def box_for_piece(piece):
    loc = piece["location_cm"]
    half = tuple(float(v) * M2CM / 2.0 for v in piece["scale"])
    return (
        piece["name"],
        (float(loc[0]) - half[0], float(loc[1]) - half[1], float(loc[2]) - half[2]),
        (float(loc[0]) + half[0], float(loc[1]) + half[1], float(loc[2]) + half[2]),
    )


def ray_box_intersection(origin, direction, bmin, bmax):
    tmin = 0.0
    tmax = float("inf")
    for axis in range(3):
        o = origin[axis]
        d = direction[axis]
        if abs(d) < 1.0e-8:
            if o < bmin[axis] or o > bmax[axis]:
                return None
            continue
        inv_d = 1.0 / d
        t0 = (bmin[axis] - o) * inv_d
        t1 = (bmax[axis] - o) * inv_d
        if t0 > t1:
            t0, t1 = t1, t0
        tmin = max(tmin, t0)
        tmax = min(tmax, t1)
        if tmax < tmin:
            return None
    return tmin if tmin > 1.0e-6 else tmax


def sunward_direction():
    yaw = math.radians(LIGHT_YAW_DEG)
    pitch = math.radians(LIGHT_PITCH_DEG)
    light_travel = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )
    return tuple(-v for v in light_travel)


def trace_sunward(pixel_x, pixel_y, *, top_wall_casts_shadow):
    origin = pixel_to_floor_world(pixel_x=pixel_x, pixel_y=pixel_y)
    direction = sunward_direction()
    shadow_casters = []
    for piece in room_pieces():
        if piece["name"] in ("floor", "window_glass"):
            continue
        if piece["name"] == "wall_y1_top" and not top_wall_casts_shadow:
            continue
        if piece["name"].startswith("window_frame_"):
            continue
        shadow_casters.append(box_for_piece(piece))

    hits = []
    for name, bmin, bmax in shadow_casters:
        t = ray_box_intersection(origin=origin, direction=direction, bmin=bmin, bmax=bmax)
        if t is not None and t > 0.0:
            hits.append((t, name))
    hits.sort()
    first = hits[0][1] if hits else "open_sky/window"
    print(
        "TRACE "
        f"pixel=({pixel_x},{pixel_y}) "
        f"world=({origin[0]:.1f},{origin[1]:.1f},{origin[2]:.1f}) "
        f"top_wall_casts_shadow={top_wall_casts_shadow} first_hit={first}",
        flush=True,
    )


def write_light_path_diagram():
    width = 1000
    height = 620
    margin = 70
    y_min = 0.0
    y_max = ROOM_SIZE_M[1] * M2CM + 40.0
    z_min = 0.0
    z_max = ROOM_SIZE_M[2] * M2CM + 30.0

    def pt(y, z):
        px = int(margin + (float(y) - y_min) / (y_max - y_min) * (width - 2 * margin))
        py = int(height - margin - (float(z) - z_min) / (z_max - z_min) * (height - 2 * margin))
        return (px, py)

    def label(image, text, y, z, color):
        cv2.putText(
            image,
            text,
            pt(y, z),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    image = np.full((height, width, 3), 245, dtype=np.uint8)
    room_y = ROOM_SIZE_M[1] * M2CM
    room_z = ROOM_SIZE_M[2] * M2CM
    window_bottom = WINDOW_Z_BOTTOM_M * M2CM
    window_top = (WINDOW_Z_BOTTOM_M + WINDOW_H_M) * M2CM

    # Room outline and back-wall pieces in Y/Z side section.
    cv2.line(image, pt(0, 0), pt(room_y, 0), (70, 70, 70), 3)
    cv2.line(image, pt(0, room_z), pt(room_y, room_z), (140, 140, 140), 2)
    cv2.line(image, pt(room_y, 0), pt(room_y, window_bottom), (40, 40, 40), 8)
    cv2.line(image, pt(room_y, window_top), pt(room_y, room_z), (40, 40, 40), 8)
    cv2.line(image, pt(room_y, window_bottom), pt(room_y, window_top), (80, 180, 255), 4)
    label(image, "floor", 10, 8, (70, 70, 70))
    label(image, "ceiling", 10, room_z - 12, (100, 100, 100))
    label(image, "window opening", room_y - 85, 160, (0, 120, 220))
    label(image, "wall_y1_top", room_y - 105, 258, (20, 20, 20))

    back_to_sun = sunward_direction()
    ratio = back_to_sun[2] / back_to_sun[1]
    samples = [
        ("shadow sample: hits wall_y1_top", 150.0, (0, 0, 220)),
        ("sun patch sample: passes window", 250.0, (0, 170, 0)),
    ]
    for text, floor_y, color in samples:
        hit_z = 0.5 + (room_y - floor_y) * ratio
        cv2.circle(image, pt(floor_y, 0.5), 6, color, -1)
        cv2.line(image, pt(floor_y, 0.5), pt(room_y, hit_z), color, 3)
        cv2.circle(image, pt(room_y, hit_z), 6, color, -1)
        label(image, text, floor_y + 8, 18, color)
        label(image, f"z@wall={hit_z:.1f}cm", room_y - 120, hit_z + 7, color)

    label(
        image,
        "If wall_y1_top does not cast shadow, the red ray becomes the new light column.",
        20,
        305,
        (0, 0, 180),
    )
    path = os.path.join(OUT_DIR, "light_path_yz.png")
    cv2.imwrite(path, image)
    print(f"DIAGRAM {path}", flush=True)


def write_contact_sheet(images, modes):
    rows = []
    for image, mode in zip(images, modes):
        if image is None:
            image = np.full((IMAGE_H, IMAGE_W, 3), 255, dtype=np.uint8)
            cv2.putText(image, mode, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        rows.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    sheet = np.vstack(rows)
    path = os.path.join(OUT_DIR, "contact.png")
    cv2.imwrite(path, sheet)
    print(f"CONTACT {path}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = [
        "physical_floor_texture",
        "physical_plain_floor",
        "directional_no_shadow",
        "top_wall_no_shadow",
    ]
    images = []
    for i, mode in enumerate(modes):
        try:
            images.append(render_mode(mode=mode, port=39820 + i))
        except Exception:
            images.append(None)
            print(f"MODE {mode} FAILED", flush=True)
            traceback.print_exc()
    write_contact_sheet(images=images, modes=modes)
    write_light_path_diagram()

    for pixel in ((535, 185), (550, 280), (685, 360), (500, 280)):
        trace_sunward(pixel_x=pixel[0], pixel_y=pixel[1], top_wall_casts_shadow=True)
        trace_sunward(pixel_x=pixel[0], pixel_y=pixel[1], top_wall_casts_shadow=False)
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
