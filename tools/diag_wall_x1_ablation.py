"""Targeted diagnostic for the "mystery checkerboard cube" artifact.

Runs 3 renders back-to-back inside a single SPEAR instance from the
FIXED camera pose (460, 390, 73) facing (yaw=-90, pitch=-11.3):

  frame_1_wall_x1_only.png     - point light + camera + wall_x1 with
                                 SetStaticMesh(Cube) + SetMaterial(MI_Floor)
                                 (5 other walls NOT spawned, floor NOT spawned)
  frame_2_wall_x1_destroyed.png - after wall_x1.destroy_actor
  frame_3_empty_smactor.png    - AStaticMeshActor spawned at (400,350,50)
                                 with NO SetStaticMesh call (tests default CDO)

Also enumerates every UPrimitiveComponent on every actor before each render
so we can spot placeholder meshes / preview components that survive our
destroy sweep.

Use spear-env:
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python \
    /data/jzy/code/SPEAR/tools/diag_wall_x1_ablation.py
"""

import os
import sys


_EXAMPLES_DIR = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + "/../examples"
)
sys.path.insert(0, _EXAMPLES_DIR)

from render_in_apartment import spawn_camera, read_frame  # noqa: E402
from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance,
    spawn_point_light,
    CUBE_MESH,
    FLOOR_MATERIAL,
    M2CM,
)


OUT_DIR = "/tmp/diag_ablation"
CAM_LOC = {"X": 460.0, "Y": 390.0, "Z": 73.0}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.3, "Yaw": -90.0}
WIDTH = 1280
HEIGHT = 720


def enumerate_primitives(game, tag):
    """Print every UPrimitiveComponent on every actor. Includes non-visible
    components (billboards, arrows, sprites, editor gizmos, reflection preview
    spheres, etc.)."""
    print(f"\n---- PRIMITIVE COMPONENTS ({tag}) ----", flush=True)
    actors = game.unreal_service.find_actors_by_class(uclass="AActor")
    total = 0
    for actor in actors:
        try:
            stable = game.unreal_service.get_stable_name_for_actor(
                actor=actor, include_unreal_name=True
            )
        except Exception:
            stable = "<no-stable>"
        try:
            prims = game.unreal_service.get_components_by_class(
                actor=actor, uclass="UPrimitiveComponent"
            )
        except Exception as e:
            prims = []
            print(f"  {stable}: <primitive enum failed: {e}>", flush=True)
            continue
        if not prims:
            continue
        for p in prims:
            total += 1
            # Try to get class name from Unreal Object; fall back to str().
            cls = "?"
            try:
                cls = p.GetClass().GetName()
            except Exception:
                try:
                    cls = str(type(p).__name__)
                except Exception:
                    pass
            visible = "?"
            hidden = "?"
            try:
                visible = str(p.IsVisible())
            except Exception:
                pass
            try:
                hidden = str(p.bHiddenInGame)
            except Exception:
                pass
            print(
                f"  {stable} | prim={cls} visible={visible} hiddenInGame={hidden}",
                flush=True,
            )
    print(f"---- {total} primitive components total ----", flush=True)


def destroy_pre_existing_renderables(game):
    """Same sweep the main script uses."""
    victims_by_class = {}
    for cls in (
        "APlayerStart", "ADefaultPawn", "ASpectatorPawn",
        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
        "ADecalActor", "AInstancedFoliageActor",
        "AGameplayDebuggerCategoryReplicator",
        "AGameplayDebuggerPlayerManager",
    ):
        try:
            victims = game.unreal_service.find_actors_by_class(uclass=cls)
        except Exception:
            victims = []
        victims_by_class[cls] = len(victims)
        for existing in victims:
            try:
                game.unreal_service.destroy_actor(actor=existing)
            except Exception:
                pass
    print(f"[destroy_pre_existing] {victims_by_class}", flush=True)


def spawn_wall_x1(game):
    """Spawn ONLY wall_x1 (east wall of a 5.2 x 4.4 x 2.8 m room),
    location + scale matches compute_shoebox_room_layout."""
    room_x_m, room_y_m, room_z_m = 5.2, 4.4, 2.8
    wall_t_m = 0.1
    rx_cm = room_x_m * M2CM
    ry_cm = room_y_m * M2CM
    rz_cm = room_z_m * M2CM
    t_cm = wall_t_m * M2CM
    loc = (rx_cm + t_cm / 2.0, ry_cm / 2.0, rz_cm / 2.0)
    scale = (wall_t_m, room_y_m, room_z_m)

    cube_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=CUBE_MESH)
    material = game.unreal_service.load_object(
        uclass="UMaterialInterface", name=FLOOR_MATERIAL
    )
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={"X": loc[0], "Y": loc[1], "Z": loc[2]},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    smc = game.unreal_service.get_component_by_class(
        actor=actor, uclass="UStaticMeshComponent"
    )
    smc.SetMobility(NewMobility="Movable")
    smc.SetStaticMesh(NewMesh=cube_mesh)
    smc.SetMaterial(ElementIndex=0, Material=material)
    actor.SetActorScale3D(
        NewScale3D={"X": scale[0], "Y": scale[1], "Z": scale[2]}
    )
    game.unreal_service.set_stable_name_for_actor(
        actor=actor, stable_name="Diag/wall_x1"
    )
    print(
        f"[spawn_wall_x1] loc={loc} scale={scale} "
        f"mesh=<Cube> mat=<MI_Floor>",
        flush=True,
    )
    return actor


def spawn_empty_static_mesh_actor(game):
    """Spawn a bare AStaticMeshActor with NO SetStaticMesh, NO SetMaterial,
    at (400, 350, 50). If the CDO / default component carries a placeholder
    mesh, it will render at this location."""
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={"X": 400.0, "Y": 350.0, "Z": 50.0},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    # Read back what the default StaticMeshComponent has, without setting anything.
    try:
        smc = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UStaticMeshComponent"
        )
        try:
            smc.SetMobility(NewMobility="Movable")
        except Exception:
            pass
        try:
            mesh = smc.GetStaticMesh()
            mesh_name = "<None>" if mesh is None else str(mesh)
        except Exception as e:
            mesh_name = f"<GetStaticMesh failed: {e}>"
        print(f"[spawn_empty_sma] default StaticMesh = {mesh_name}", flush=True)
    except Exception as e:
        print(f"[spawn_empty_sma] no StaticMeshComponent? {e}", flush=True)
    try:
        game.unreal_service.set_stable_name_for_actor(
            actor=actor, stable_name="Diag/empty_sma"
        )
    except Exception:
        pass
    return actor


def render_one(instance, cam, comp, out_name):
    import cv2

    # Move camera to fixed pose, let scene settle, capture.
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(
            NewLocation=CAM_LOC,
            NewRotation=CAM_ROT,
            bSweep=False,
            bTeleport=True,
        )
    with instance.end_frame():
        pass
    instance.step(num_frames=12)
    with instance.begin_frame():
        pass
    with instance.end_frame():
        frame = read_frame(comp)
        path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(path, frame)
    print(f"[render] wrote {path}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        # ---- Setup: destroy pre-existing renderables, spawn point light,
        # spawn camera, spawn wall_x1 alone.
        with instance.begin_frame():
            destroy_pre_existing_renderables(game=game)
            spawn_point_light(
                game=game,
                x_cm=260.0, y_cm=220.0, z_cm=265.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            wall = spawn_wall_x1(game=game)
            cam, comp = spawn_camera(game=game, width=WIDTH, height=HEIGHT)
        with instance.end_frame():
            pass
        instance.step(num_frames=4)

        # Enumerate primitive components BEFORE any render, so we can spot
        # anything besides wall_x1 that is still hanging around.
        with instance.begin_frame():
            enumerate_primitives(game=game, tag="wall_x1 spawned")
        with instance.end_frame():
            pass

        # ---- Frame 1: wall_x1 alone.
        render_one(instance, cam, comp, "frame_1_wall_x1_only.png")

        # ---- Frame 2: destroy wall_x1, render again.
        with instance.begin_frame():
            try:
                game.unreal_service.destroy_actor(actor=wall)
                print("[destroy] wall_x1 destroyed", flush=True)
            except Exception as e:
                print(f"[destroy] wall_x1 destroy failed: {e}", flush=True)
        with instance.end_frame():
            pass
        instance.step(num_frames=4)
        with instance.begin_frame():
            enumerate_primitives(game=game, tag="wall_x1 destroyed")
        with instance.end_frame():
            pass
        render_one(instance, cam, comp, "frame_2_wall_x1_destroyed.png")

        # ---- Frame 3: spawn bare AStaticMeshActor with NO SetStaticMesh,
        # at (400, 350, 50). Tests default CDO placeholder.
        with instance.begin_frame():
            empty = spawn_empty_static_mesh_actor(game=game)
        with instance.end_frame():
            pass
        instance.step(num_frames=4)
        with instance.begin_frame():
            enumerate_primitives(game=game, tag="empty AStaticMeshActor spawned")
        with instance.end_frame():
            pass
        render_one(instance, cam, comp, "frame_3_empty_smactor.png")
        _ = empty
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
