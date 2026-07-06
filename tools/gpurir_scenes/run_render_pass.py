"""Render a scene into one room with 4 fixed camera yaws.

room="apartment"  -> apartment_0000 map via configure_instance()
room="shoebox"    -> shoebox 5.2x4.4x2.8 via configure_gpurir_instance() + _spawn_room_and_lights()

Camera sits at world-space mic position (2.6, 2.2, 1.2) [meters]. Yaws swept:
{0, 90, 180, 270}. Yaw=0 points +X in UE. Our SceneSpec convention has
mic-forward = +Y-world = window direction. We align UE +X to world +Y by
rotating actor positions -90 degrees (via a rotate transform) OR, simpler,
we skip that rotation and just note in trajectory.json that
"camera yaw=0 UE" == "camera facing +X UE" == "mic-forward-world".
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import types

import cv2
import numpy as np
import spear

REPO = "/data/jzy/code/SPEAR"
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from render_in_apartment import (  # noqa: E402
    APARTMENT_MAP, configure_instance, spawn_camera, read_frame,
)
from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance, spawn_sky, spawn_directional_light, spawn_point_light,
    compute_shoebox_room_layout,
    resolve_floor_material, resolve_wall_material,
    piece_casts_shadow, spawn_room_piece,
)
from gpurir_scenes.scene_spec import compose_scene, check_no_clipping, N_FRAMES, FPS, MIC_POS_M  # noqa: E402
from gpurir_scenes.furniture_map import load_apartment_furniture  # noqa: E402


M2CM = 100.0
WIDTH = 640
HEIGHT = 480
# Wider FoV pulls the whole room into frame — animals look further from the
# camera (no "vertigo effect") and edge-of-frame subjects like a static dog
# at the -X side of the room stop getting head-clipped by the frustum edge.
# 120 H keeps long, side-facing quadrupeds away from the horizontal frustum
# edge while still preserving a room-scale perspective.
CAMERA_FOV_DEG = 120.0

# CAMERA YAWS per room.
#
# SceneSpec convention: mic-forward = world +Y = window direction.
#
# PER-ROOM YAWS:
#   Shoebox: room stays in its native +Y half (0..520, 0..440). Actors go
#     there too (no Y flip). Camera view0 UE yaw=90 (facing +Y = front wall).
#     Frame R in camera space = UE -X = world -X. So we horizontally-flip
#     shoebox frames after render so frame R visually = world +X, matching
#     the audio pan convention.
#   Apartment: window is on UE -Y. Camera view0 UE yaw=-90 (facing -Y).
#     Actor coords use Y flip. Frame R = UE +X = world +X. No hflip needed.
_YAW_WORLD_TO_UE_SHOEBOX = 90.0
_YAW_WORLD_TO_UE_APARTMENT = -90.0


def _camera_yaws(room):
    base = _YAW_WORLD_TO_UE_SHOEBOX if room == "shoebox" else _YAW_WORLD_TO_UE_APARTMENT
    return [base + off for off in (0.0, 90.0, 180.0, 270.0)]


def _set_capture_fov(comp):
    comp.set_property_value(property_name="FOVAngle", property_value=CAMERA_FOV_DEG)

# STATIC ROLL: Hunyuan Flux mesh convention is Y-up (side-profile Flux render
# put "up" along Y). UE is Z-up. Rotate around X by -90 to align Y-up -> Z-up.
# (Roll=+90 rotates the WRONG WAY -- head goes below the floor.)
_STATIC_ROLL_DEG = -90.0
_STATIC_ADDITIONAL_YAW_DEG = 0.0

# In the shoebox room, world origin is the room corner (0,0,0), so mic and
# actor positions from SceneSpec (which are in meters from origin) map
# directly: X_ue_cm = x_m * 100.
#
# apartment_0000 map has its own arbitrary world origin. We treat SceneSpec
# coordinates as OFFSETS from an apartment-local mic anchor. The Clock spawn
# point (validated ground-clean) is the anchor.
APARTMENT_MIC_ORIGIN_CM = (-120.0, 80.0, 120.0)  # x_cm, y_cm, z_cm
APARTMENT_FLOOR_Z_CM = 27.1  # ground trace at Clock spawn


def _bp_path(placement):
    if placement.is_animated:
        return f"/Game/MyAssets/Audioset/Blueprints/gate_{placement.tag}/BP_gate_{placement.tag}.BP_gate_{placement.tag}_C"
    return f"/Game/MyAssets/Audioset/Blueprints/gate_static_{placement.tag}/BP_gate_static_{placement.tag}.BP_gate_static_{placement.tag}_C"


def _spawn_shoebox(game, room_size_m):
    """Minimal shoebox: 4 walls, floor, ceiling with default materials.
    Skip the fancy window/glass — this is just a visual container for the actor.
    """
    # Passing None uses deterministic pool index; explicit label strings are
    # NOT UE material paths and cause load_object to fail.
    resolved_floor = resolve_floor_material(floor_material=None, floor_material_seed=0)
    resolved_wall = resolve_wall_material(wall_material=None, wall_material_seed=0)
    pieces = compute_shoebox_room_layout(room_size_m=room_size_m, wall_thickness_m=0.1)
    for piece in pieces:
        mat = resolved_wall
        if piece["name"] == "floor":
            mat = resolved_floor
        spawn_room_piece(
            game=game, piece=piece, material_path=mat,
            cast_shadow=piece_casts_shadow(
                name=piece["name"],
                ceiling_casts_shadow=False,
                window_top_wall_casts_shadow=True,
                window_wall_casts_shadow=True,
            ),
        )
    spawn_sky(game=game)
    spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-45.0, intensity_lux=8.0)
    rx, ry, rz = (v * M2CM for v in room_size_m)
    spawn_point_light(game=game, x_cm=rx / 2, y_cm=ry / 2, z_cm=rz - 30,
                      intensity_lumens=6000.0, attenuation_cm=800.0)


def _world_from_scene(pos_m, room, spec, actor_z_lift_cm=0.0):
    """Map SceneSpec position (meters, world coords) to UE cm.

    Shoebox: no Y flip; mic-forward=world+Y=UE+Y. Room untouched.
    Apartment: Y flipped (world+Y=UE-Y).
    """
    x_m, y_m, _z_m = pos_m
    mic_x_m, mic_y_m, _ = spec.mic_pos_m
    dx_cm = (x_m - mic_x_m) * M2CM
    if room == "shoebox":
        dy_cm = (y_m - mic_y_m) * M2CM
        return (mic_x_m * M2CM + dx_cm, mic_y_m * M2CM + dy_cm, actor_z_lift_cm)
    dy_cm = -(y_m - mic_y_m) * M2CM
    return (
        APARTMENT_MIC_ORIGIN_CM[0] + dx_cm,
        APARTMENT_MIC_ORIGIN_CM[1] + dy_cm,
        APARTMENT_FLOOR_Z_CM + actor_z_lift_cm,
    )


def _yaw_world_to_ue(yaw_world_deg, room):
    return yaw_world_deg if room == "shoebox" else -yaw_world_deg


# After Roll=90 around X, mesh Y-axis becomes world Z. Z lift = half of
# original Y-extent, so feet touch the floor. Measured from .glb bboxes.
_STATIC_Z_LIFT_BY_TAG_CM = {
    "pig":            32.0,   # Y-extent 0.63m
    "horse":          71.0,   # Y-extent 1.42m
    "sheep":          60.0,   # Y-extent 1.19m
    "yak":            44.0,   # Y-extent 0.87m
    "goat":           23.0,   # Y-extent 0.45m
    "donkey_ass":     64.0,   # Y-extent 1.28m
    "cattle_bovinae": 38.0,   # Y-extent 0.76m
}
_ANIMATED_Z_LIFT_CM = 0.0


def _play_anim_on_actor(game, actor, placement):
    """Swap the SkeletalMeshComponent's playing anim to wanted_anim.

    The BP's default is Walking; if wanted_anim is Idle, we look for an
    UAnimationAsset ending in /Idle under the mesh dir and PlayAnimation
    it. If we can't find one we leave the default (Walking).
    """
    smc = game.unreal_service.get_component_by_class(
        actor=actor, uclass="USkeletalMeshComponent",
    )
    if smc is None:
        return
    smc.SetComponentTickEnabled(bEnabled=True)
    wanted = getattr(placement, "wanted_anim", "Walking") or "Walking"
    if wanted == "Walking":
        # BP already set to Walking on import; nothing to do.
        return
    anim_path = f"/Game/MyAssets/Audioset/Meshes/gate_{placement.tag}/{wanted}"
    try:
        anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=anim_path)
        smc.PlayAnimation(NewAnimToPlay=anim, bLooping=True)
    except Exception as e:
        print(f"[render] WARN could not swap anim to {wanted} for {placement.tag}: {e}", flush=True)


def _spawn_animal(game, placement, room, spec):
    bp = game.unreal_service.load_class(uclass="AActor", name=_bp_path(placement))
    if placement.is_animated:
        p0 = placement.trajectory_m[0]
        z_lift = _ANIMATED_Z_LIFT_CM
    else:
        p0 = placement.static_pos_m
        z_lift = _STATIC_Z_LIFT_BY_TAG_CM.get(placement.tag, 50.0)
    x_cm, y_cm, z_cm = _world_from_scene(p0, room, spec, actor_z_lift_cm=z_lift)
    actor = game.unreal_service.spawn_actor(
        uclass=bp,
        location={"X": x_cm, "Y": y_cm, "Z": z_cm},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    # Scale tuning:
    #   Animated: 0.15 -> ~0.7m tall dog (real medium-large dog size).
    #     Previously 0.3 gave 1.4m tall dogs (way too big).
    #   Static Hunyuan meshes: 1.0 (Interchange glTF converts meters->cm,
    #     so a 1.7m sheep mesh becomes 170 UE-cm at scale 1.0).
    scale = 0.15 if placement.is_animated else 1.0
    actor.SetActorScale3D(NewScale3D={"X": scale, "Y": scale, "Z": scale})
    if not placement.is_animated:
        # Static Hunyuan meshes are Y-up (Flux side-profile); need Roll=-90
        # to align Y-axis with UE +Z. Yaw uses room-appropriate flip.
        yaw_world = float(placement.static_yaw_deg) + _STATIC_ADDITIONAL_YAW_DEG
        actor.K2_SetActorLocationAndRotation(
            NewLocation={"X": x_cm, "Y": y_cm, "Z": z_cm},
            NewRotation={"Roll": _STATIC_ROLL_DEG,
                          "Pitch": 0.0,
                          "Yaw": _yaw_world_to_ue(yaw_world, room)},
            bSweep=False, bTeleport=True,
        )
    else:
        # For rig-based animals, honor the placement's wanted_anim so a
        # "stationary" dog can be shown as Idle instead of the default Walking.
        _play_anim_on_actor(game, actor, placement)
    return actor


def _step_animated(actor, placement, frame_i, room, spec):
    p = placement.trajectory_m[frame_i]
    y_world = float(placement.yaw_deg[frame_i])
    x_cm, y_cm, z_cm = _world_from_scene(p, room, spec, actor_z_lift_cm=_ANIMATED_Z_LIFT_CM)
    actor.K2_SetActorLocationAndRotation(
        NewLocation={"X": x_cm, "Y": y_cm, "Z": z_cm},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": _yaw_world_to_ue(y_world, room)},
        bSweep=False, bTeleport=True,
    )


def run_render_pass(spec, room, out_dir):
    assert room in ("apartment", "shoebox")
    os.makedirs(out_dir, exist_ok=True)

    # Apartment renders must respect furniture bboxes. Even when the spec came
    # from compose_scene (which already avoids furniture), hand-authored specs
    # (e.g. scene_two_dogs) skip that path -- so double-check here before we
    # spend UE render time on a clipping scene.
    if room == "apartment":
        try:
            furniture_bboxes = load_apartment_furniture()
        except Exception as e:
            print(f"[render] WARN could not load apartment furniture map: {e}",
                  flush=True)
            furniture_bboxes = None
        if furniture_bboxes:
            check_no_clipping(spec, furniture_bboxes=furniture_bboxes)

    if room == "apartment":
        instance = configure_instance(rpc_port=39002)
    else:
        instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()

    try:
        with instance.begin_frame():
            if room == "shoebox":
                _spawn_shoebox(game, spec.room_size_m)
            # For apartment: keep native lighting/geometry. Optional cleanup of
            # spawned pawns/actors if needed. We spawn nothing else -- animals
            # + camera go on top of native scene.
            cam, comp = spawn_camera(game=game, width=WIDTH, height=HEIGHT)
            # Widen FoV to CAMERA_FOV_DEG so animals visually recede and the
            # dog at frame edge stops getting head-clipped. spawn_camera set
            # FOVAngle from viewport (usually ~90); override on the SceneCapture
            # component that actually renders the pixels we read.
            try:
                _set_capture_fov(comp)
            except Exception as e:
                print(f"[render] WARN could not set FOVAngle: {e}", flush=True)
            actors = [_spawn_animal(game, a, room, spec) for a in spec.animals]
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        # Warm up long enough for virtual-texture streaming to bring all
        # floor/wall/mesh textures to full resolution BEFORE we sample the
        # first frame. 20 was too short (user saw ~3s of blurry floor).
        instance.step(num_frames=120)

        # Mic UE position: shoebox = identity, apartment = fixed anchor.
        if room == "shoebox":
            mic_x_cm = spec.mic_pos_m[0] * M2CM
            mic_y_cm = spec.mic_pos_m[1] * M2CM
            mic_z_cm = spec.mic_pos_m[2] * M2CM
        else:
            mic_x_cm, mic_y_cm, mic_z_cm = APARTMENT_MIC_ORIGIN_CM

        camera_yaws = _camera_yaws(room)

        # For each of 4 fixed yaws, render N_FRAMES with camera fixed at mic
        # and (for animated actors) animated actors stepping along their traj.
        for view_i, yaw in enumerate(camera_yaws):
            # Fresh warmup for each yaw so streaming catches up with the new
            # visible surfaces before frame 0 is sampled.
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                    NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw)},
                    bSweep=False, bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=40)
            for frame_i in range(N_FRAMES):
                with instance.begin_frame():
                    for actor, placement in zip(actors, spec.animals):
                        if placement.is_animated:
                            _step_animated(actor, placement, frame_i, room, spec)
                    cam.K2_SetActorLocationAndRotation(
                        NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw)},
                        bSweep=False, bTeleport=True,
                    )
                with instance.end_frame():
                    img = read_frame(comp)
                    cv2.imwrite(os.path.join(out_dir, f"view{view_i}_frame_{frame_i:04d}.png"), img)
            out_mp4 = os.path.join(out_dir, f"view{view_i}.mp4")
            # Shoebox camera (yaw=+Y_UE) has visual right = -X_world; horizontally
            # flip so visual right = +X_world = audio right channel.
            vf_args = ["-vf", "hflip"] if room == "shoebox" else []
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", os.path.join(out_dir, f"view{view_i}_frame_%04d.png"),
                *vf_args,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4,
            ], check=True, capture_output=True)
    finally:
        instance.close(force=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--room", required=True, choices=["apartment", "shoebox"])
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # For apartment scenes, pass furniture to compose_scene so sampling avoids
    # furniture from the start (not just at the final check).
    furniture = load_apartment_furniture() if args.room == "apartment" else None
    spec = compose_scene(seed=args.seed, furniture_bboxes=furniture)
    run_render_pass(spec, args.room, args.out_dir)
    print(f"RENDER_DONE {args.out_dir}")
