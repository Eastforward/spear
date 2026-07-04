"""Render an animated Quaternius dog inside the SPEAR shoebox room, driven
along either a GPURIR-style random trajectory or a user waypoint list.

Spec: docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md
Plan: docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md

USAGE (GPURIR trajectory):
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \\
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
    examples/render_animated_dog_gpurir.py \\
    --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \\
    --run-name animated_dog_gpurir_seed42

USAGE (waypoint override):
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \\
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
    examples/render_animated_dog_gpurir.py \\
    --trajectory-mode waypoints \\
    --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \\
    --run-name animated_dog_waypoint_Lshape
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# copied verbatim from render_in_gpurir_room.py::render_gpurir_room — BEGIN
from render_in_gpurir_room import (  # noqa: E402
    WALL_JOINT_OVERLAP_M,
    WALL_THICKNESS_M,
    FLOOR_MESH,
    FLOOR_MESH_TILE_M,
    FLOOR_MESH_THICKNESS_CM,
    FLOOR_MATERIAL_POOL,
    WALL_MATERIAL_POOL,
    FLOOR_MATERIAL,
    WALL_MATERIAL,
    GLASS_MATERIAL,
    WINDOW_FRAME_MATERIAL,
    WINDOW_FRAME_THICKNESS_M,
    GLASS_THICKNESS_M,
    OUTDOOR_GROUND_MATERIAL,
    CUBE_MESH,
    EMPTY_MAP,
    SPEARSIM_EXECUTABLE,
    DEFAULT_TMP_ROOT,
    M2CM,
    compute_shoebox_room_layout,
    compute_window_wall_layout,
    compute_window_frame_layout,
    spawn_room_piece,
    spawn_directional_light,
    spawn_point_light,
    spawn_sphere_reflection_capture,
    spawn_sky,
    _material_for_piece,
    _try_hide,
    piece_casts_shadow,
    resolve_floor_material,
    resolve_wall_material,
    configure_gpurir_instance,
)
# copied verbatim from render_in_gpurir_room.py::render_gpurir_room — END

from render_in_apartment import spawn_camera, read_frame, clean_frames  # noqa: E402
from trajectory import (  # noqa: E402
    gpurir_trajectory,
    waypoint_trajectory,
    compute_yaw_from_positions,
)


ANIMATED_DOG_BP = "/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C"
ANIMATED_DOG_WALKING_ANIM = "/Game/MyAssets/Audioset/Meshes/animated_dog/Walking"
DEFAULT_RENDER_ROOT = "/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir"


def parse_waypoints(s):
    """Parse '0.5,0.5;2.6,0.5;2.6,4.0' into [(0.5,0.5), (2.6,0.5), (2.6,4.0)]."""
    out = []
    for piece in s.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        parts = [float(x) for x in piece.split(",")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"waypoint must be x,y or x,y,z (got {piece!r})"
            )
        out.append(tuple(parts))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # trajectory
    p.add_argument("--trajectory-mode", choices=["gpurir", "waypoints"], default="gpurir")
    p.add_argument("--trajectory-seed", type=int, default=0)
    p.add_argument("--speed-bucket", choices=["A", "B", "C", "D"], default="B")
    p.add_argument("--waypoints", type=parse_waypoints, default=None,
                   help="Semicolon-separated waypoints, e.g. '0.5,0.5;2.6,4.0'")
    p.add_argument("--source-height-m", type=float, default=0.45)
    p.add_argument("--z-offset-m", type=float, default=0.0,
                   help="Vertical offset applied to every frame (metres). "
                        "Set negative if the mesh pivot is above the foot.")
    p.add_argument("--dog-scale", type=float, default=0.12,
                   help="Uniform actor scale applied to BP_dog_animated. "
                        "Quaternius Dog.glb has native extent ~6.9m (much larger "
                        "than a real dog); 0.12 * 690cm ~= 80cm tall dog.")
    # room + lights (defaults chosen to match the last static-dog run visually)
    p.add_argument("--room-size-m", type=float, nargs=3, default=[5.2, 4.4, 2.8],
                   metavar=("X", "Y", "Z"))
    p.add_argument("--wall-thickness-m", type=float, default=WALL_THICKNESS_M)
    p.add_argument("--window-w-m", type=float, default=1.4)
    p.add_argument("--window-h-m", type=float, default=1.4)
    p.add_argument("--window-frame-thickness-m", type=float, default=WINDOW_FRAME_THICKNESS_M)
    p.add_argument("--window-cx-m", type=float, default=None,
                   help="Window center X (m). Default = room_x/2.")
    p.add_argument("--window-z-bottom-m", type=float, default=0.9)
    p.add_argument("--floor-material", default=None)
    p.add_argument("--floor-material-seed", type=int, default=0)
    p.add_argument("--wall-material", default=None)
    p.add_argument("--wall-material-seed", type=int, default=0)
    p.add_argument("--ceiling-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window-top-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ceiling-light-lumens", type=float, default=6000.0)
    p.add_argument("--ceiling-light-drop-cm", type=float, default=15.0)
    p.add_argument("--ceiling-light-attenuation-cm", type=float, default=600.0)
    p.add_argument("--reflection-capture", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--directional-light-yaw-deg", type=float, default=-90.0)
    p.add_argument("--directional-light-pitch-deg", type=float, default=-30.0)
    p.add_argument("--directional-light-intensity-lux", type=float, default=10.0)
    # camera
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    # frames
    p.add_argument("--n-frames", type=int, default=36)
    p.add_argument("--framerate", type=int, default=12)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--per-frame-warmup-frames", type=int, default=6)
    # output
    p.add_argument("--rpc-port", type=int, default=39002)
    p.add_argument("--output-root", default=DEFAULT_RENDER_ROOT)
    p.add_argument("--run-name", required=True)
    args = p.parse_args(argv)
    if args.trajectory_mode == "waypoints" and not args.waypoints:
        p.error("--trajectory-mode waypoints requires --waypoints '...'")
    return args


def _room_layout(args):
    """Compute pre-args room layout — identical shape to render_in_gpurir_room."""
    room_pieces = compute_shoebox_room_layout(
        room_size_m=args.room_size_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_cx = args.window_cx_m if args.window_cx_m is not None else args.room_size_m[0] / 2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_frame_pieces = compute_window_frame_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
        frame_thickness_m=args.window_frame_thickness_m,
    )
    non_y1 = [p for p in room_pieces if p["name"] != "wall_y1"]

    ry_cm = args.room_size_m[1] * M2CM
    t_cm = args.wall_thickness_m * M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx * M2CM,
            ry_cm + t_cm / 2.0,
            (args.window_z_bottom_m + args.window_h_m / 2.0) * M2CM,
        ),
        "scale": (args.window_w_m, GLASS_THICKNESS_M, args.window_h_m),
    }
    outdoor_ground_piece = {
        "name": "outdoor_ground",
        "location_cm": (
            args.room_size_m[0] * M2CM / 2.0,
            args.room_size_m[1] * M2CM / 2.0,
            -args.wall_thickness_m * M2CM - 5.0,
        ),
        "scale": (80.0, 80.0, 0.1),
    }
    return non_y1 + window_pieces + window_frame_pieces + [glass_piece, outdoor_ground_piece]


def _spawn_room_and_lights(args, game):
    """Room+light spawn sequence copied from render_in_gpurir_room.py."""
    resolved_floor = resolve_floor_material(
        floor_material=args.floor_material,
        floor_material_seed=args.floor_material_seed,
    )
    resolved_wall = resolve_wall_material(
        wall_material=args.wall_material,
        wall_material_seed=args.wall_material_seed,
    )
    print(f"[render_animated_dog] floor={resolved_floor}", flush=True)
    print(f"[render_animated_dog] wall={resolved_wall}", flush=True)

    all_pieces = _room_layout(args)
    for piece in all_pieces:
        mat = _material_for_piece(piece["name"], wall_material=resolved_wall)
        if piece["name"] == "floor":
            mat = resolved_floor
        print(f"[render_animated_dog] spawning piece={piece['name']} mat={mat}", flush=True)
        spawn_room_piece(
            game=game,
            piece=piece,
            material_path=mat,
            cast_shadow=piece_casts_shadow(
                name=piece["name"],
                ceiling_casts_shadow=args.ceiling_casts_shadow,
                window_top_wall_casts_shadow=args.window_top_wall_casts_shadow,
                window_wall_casts_shadow=args.window_wall_casts_shadow,
            ),
        )

    spawn_sky(game=game)
    spawn_directional_light(
        game=game,
        yaw_deg=args.directional_light_yaw_deg,
        pitch_deg=args.directional_light_pitch_deg,
        intensity_lux=args.directional_light_intensity_lux,
    )
    room_x_cm = args.room_size_m[0] * M2CM
    room_y_cm = args.room_size_m[1] * M2CM
    room_z_cm = args.room_size_m[2] * M2CM
    spawn_point_light(
        game=game,
        x_cm=room_x_cm / 2.0,
        y_cm=room_y_cm / 2.0,
        z_cm=room_z_cm - args.ceiling_light_drop_cm,
        intensity_lumens=args.ceiling_light_lumens,
        attenuation_cm=args.ceiling_light_attenuation_cm,
    )
    if args.reflection_capture:
        spawn_sphere_reflection_capture(
            game=game,
            x_cm=room_x_cm / 2.0,
            y_cm=room_y_cm / 2.0,
            z_cm=room_z_cm / 2.0,
            influence_radius_cm=max(room_x_cm, room_y_cm, room_z_cm),
        )
    return resolved_floor, resolved_wall


def _spawn_animated_dog(game, x_cm, y_cm, z_cm, scale=0.12):
    """Spawn BP_dog_animated and start Walking on loop. Returns (actor, smc).

    Uses PlayAnimation directly (single BlueprintCallable UFUNCTION that
    internally SetAnimationMode+SetAnimation+Play). UE 5.5 native param
    names: `NewAnimToPlay` (UAnimationAsset*) + `bLooping` (bool).
    Reference: UE 5.5 SkeletalMeshComponent.h:1126.

    scale: uniform actor scale to correct the Quaternius Dog.glb native
    ~6.9m extent down to something dog-sized.

    Tick fix: SkeletalMeshActor + SkeletalMeshComponent don't tick every
    frame in cooked/game builds by default, so the anim clock never
    advances (legs freeze in T-pose/first-frame). We explicitly enable
    Tick on both, and additionally call SetPlayRate + SetPosition so the
    animation cursor is guaranteed to move.
    """
    bp = game.unreal_service.load_class(uclass="AActor", name=ANIMATED_DOG_BP)
    actor = game.unreal_service.spawn_actor(
        uclass=bp,
        location={"X": float(x_cm), "Y": float(y_cm), "Z": float(z_cm)},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    actor.SetActorScale3D(
        NewScale3D={"X": float(scale), "Y": float(scale), "Z": float(scale)}
    )
    # Force actor + component ticking (cooked builds default the SkeletalMeshActor
    # to a slow update mode that freezes anim after the first frame).
    try:
        actor.SetActorTickEnabled(bEnabled=True)
    except Exception as e:
        print(f"[render_animated_dog] SetActorTickEnabled warn: {e}", flush=True)

    smc = game.unreal_service.get_component_by_class(actor=actor, uclass="USkeletalMeshComponent")
    try:
        smc.SetComponentTickEnabled(bEnabled=True)
    except Exception as e:
        print(f"[render_animated_dog] SetComponentTickEnabled warn: {e}", flush=True)
    # Make sure the SkeletalMesh update mode is 'Always Tick Pose and Refresh Bones'
    # so the pose evaluates every frame (cooked default is often
    # OnlyTickPoseWhenRendered which can skip if we're between frames).
    try:
        smc.SetVisibilityBasedAnimTickOption(
            NewVisibilityBasedAnimTickOption="AlwaysTickPoseAndRefreshBones"
        )
    except Exception as e:
        print(f"[render_animated_dog] SetVisibilityBasedAnimTickOption warn: {e}", flush=True)

    anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=ANIMATED_DOG_WALKING_ANIM)
    smc.PlayAnimation(NewAnimToPlay=anim, bLooping=True)
    # Belt-and-suspenders: ensure the anim clock actually advances.
    try:
        smc.SetPlayRate(Rate=1.0)
    except Exception:
        pass
    return actor, smc


def _compute_trajectory(args):
    if args.trajectory_mode == "gpurir":
        pos_m = gpurir_trajectory(
            room_size_m=tuple(args.room_size_m),
            n_frames=args.n_frames,
            speed_bucket=args.speed_bucket,
            source_height_m=args.source_height_m,
            traj_aug=True,
            seed=args.trajectory_seed,
            traj_pts_full=200,
        )
    else:
        pos_m = waypoint_trajectory(
            waypoints_m=args.waypoints,
            n_frames=args.n_frames,
            room_size_m=tuple(args.room_size_m),
            source_height_m=args.source_height_m,
            kind="cubic",
        )
    if args.z_offset_m:
        pos_m[:, 2] = pos_m[:, 2] + args.z_offset_m
    yaw_deg = compute_yaw_from_positions(pos_m)
    return pos_m, yaw_deg


def render_animated_dog(args):
    output_dir = os.path.join(args.output_root, args.run_name)
    clean_frames(output_dir)
    positions_m, yaw_deg = _compute_trajectory(args)

    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # Defensive sweep of Entry-map defaults + spawn-cube prevention
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
                        "ADecalActor", "AInstancedFoliageActor",
                        "AGameplayDebuggerCategoryReplicator",
                        "AGameplayDebuggerPlayerManager"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            resolved_floor, resolved_wall = _spawn_room_and_lights(args, game)
            print("[render_animated_dog] room+lights done, spawning camera...", flush=True)
            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            # Fixed camera pose: park it near one corner of the room, looking
            # diagonally across so both the mic center and the moving source are
            # generally in-frame. Static-dog script instead ORBITS the camera
            # around the source, but for a moving dog we want a static viewpoint
            # so the dog's motion is visible relative to the room.
            rx_cm = args.room_size_m[0] * M2CM
            ry_cm = args.room_size_m[1] * M2CM
            rz_cm = args.room_size_m[2] * M2CM
            cam_x = 0.5 * M2CM   # 0.5m from the -x wall
            cam_y = 0.5 * M2CM   # 0.5m from the -y wall
            cam_z = 1.6 * M2CM   # eye height
            # Aim at room center (mic position)
            import math as _m
            look_x, look_y, look_z = rx_cm / 2.0, ry_cm / 2.0, 1.2 * M2CM
            # Named cam_yaw_deg / cam_pitch_deg to avoid shadowing the yaw_deg
            # ARRAY returned by _compute_trajectory (which drives the dog per
            # frame). Silent shadow would blow up later at yaw_deg[i].
            cam_yaw_deg = _m.degrees(_m.atan2(look_y - cam_y, look_x - cam_x))
            cam_pitch_deg = -_m.degrees(_m.atan2(cam_z - look_z,
                                                 _m.hypot(look_x - cam_x, look_y - cam_y)))
            cam.K2_SetActorLocationAndRotation(
                NewLocation={"X": cam_x, "Y": cam_y, "Z": cam_z},
                NewRotation={"Roll": 0.0, "Pitch": cam_pitch_deg, "Yaw": cam_yaw_deg},
                bSweep=False, bTeleport=True,
            )
            print(f"[render_animated_dog] camera at ({cam_x:.0f},{cam_y:.0f},{cam_z:.0f}) "
                  f"yaw={cam_yaw_deg:.1f} pitch={cam_pitch_deg:.1f}, spawning BP_dog_animated...", flush=True)
            # Actor spawns AT FLOOR (z=0) — positions_m[:,2] is the
            # audio-side source height (dog mouth ~0.45m), NOT the actor
            # transform Z. Using source height for actor Z made the dog
            # visually float 45cm above the floor.
            dog_actor, _dog_smc = _spawn_animated_dog(
                game,
                x_cm=positions_m[0, 0] * M2CM,
                y_cm=positions_m[0, 1] * M2CM,
                z_cm=args.z_offset_m * M2CM,
                scale=args.dog_scale,
            )
            print("[render_animated_dog] dog spawned OK", flush=True)
        with instance.end_frame():
            pass

        # VT warmup
        instance.step(num_frames=args.warmup_frames)

        # Per-frame drive: teleport dog, wait a few frames for the walk
        # animation to advance, render.
        for i in range(args.n_frames):
            with instance.begin_frame():
                dog_actor.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": float(positions_m[i, 0] * M2CM),
                        "Y": float(positions_m[i, 1] * M2CM),
                        # actor stays on the floor (z_offset_m default 0);
                        # audio-side height is metadata-only, not the actor Z
                        "Z": float(args.z_offset_m * M2CM),
                    },
                    NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_deg[i])},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=args.per_frame_warmup_frames)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{i:04d}.png"),
                    read_frame(comp),
                )

        # ffmpeg mux
        video_path = os.path.join(output_dir, "turntable.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.framerate),
                "-i", os.path.join(output_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                video_path,
            ],
            check=True, capture_output=True,
        )
        print(f"VIDEO_DONE {video_path}", flush=True)

        # trajectory.json — cross-modal alignment metadata
        traj_json = {
            "trajectory_mode": args.trajectory_mode,
            "trajectory_seed": args.trajectory_seed,
            "speed_bucket": args.speed_bucket,
            "source_height_m": args.source_height_m,
            "z_offset_m": args.z_offset_m,
            "room_size_m": list(args.room_size_m),
            "n_frames": args.n_frames,
            "traj_pts_full": 200,
            "positions_m": positions_m.tolist(),
            "yaw_deg": yaw_deg.tolist(),
            "mic_pos_m": [args.room_size_m[0] / 2.0, args.room_size_m[1] / 2.0, 1.2],
            "resolved_floor_material": resolved_floor,
            "resolved_wall_material": resolved_wall,
        }
        if args.trajectory_mode == "waypoints":
            traj_json["waypoints_m"] = [list(wp) for wp in args.waypoints]
        with open(os.path.join(output_dir, "trajectory.json"), "w") as f:
            json.dump(traj_json, f, indent=2)
        print(f"TRAJECTORY_JSON_DONE {os.path.join(output_dir, 'trajectory.json')}", flush=True)
    finally:
        instance.close(force=True)


def main(argv=None):
    args = parse_args(argv)
    render_animated_dog(args)


if __name__ == "__main__":
    main()
