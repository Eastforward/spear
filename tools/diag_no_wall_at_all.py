"""Follow-up diagnostic. NEVER spawn wall_x1. Just:
  - destroy pre-existing renderables
  - spawn point light
  - spawn camera
  - render one frame from (460, 390, 73) facing (-90, -11.3)

If the checkerboard cube STILL appears in this frame, then it is
categorically NOT related to wall_x1 or to any AStaticMeshActor we spawn.
Then also enumerate EVERY UObject class subclass in the scene, not just AActor.
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
)


OUT_DIR = "/tmp/diag_ablation"
CAM_LOC = {"X": 460.0, "Y": 390.0, "Z": 73.0}
CAM_ROT = {"Roll": 0.0, "Pitch": -11.3, "Yaw": -90.0}


def destroy_pre_existing_renderables(game):
    victims_by_class = {}
    for cls in (
        "APlayerStart", "ADefaultPawn", "ASpectatorPawn",
        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
        "ADecalActor", "AInstancedFoliageActor",
        "AGameplayDebuggerCategoryReplicator",
        "AGameplayDebuggerPlayerManager",
        # Try MORE classes to see what else exists
        "AReflectionCapture", "ASphereReflectionCapture",
        "ABoxReflectionCapture", "APlaneReflectionCapture",
        "AVolumetricCloud", "APostProcessVolume",
        "ANavigationData", "ANavMeshBoundsVolume",
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


def enumerate_all_actors_with_primitives(game, tag):
    print(f"\n---- ALL ACTORS + ANY UPrimitiveComponents ({tag}) ----", flush=True)
    actors = game.unreal_service.find_actors_by_class(uclass="AActor")
    print(f"total AActor count = {len(actors)}", flush=True)
    for actor in actors:
        try:
            stable = game.unreal_service.get_stable_name_for_actor(
                actor=actor, include_unreal_name=True
            )
        except Exception:
            stable = "<no-stable>"
        # ANY prim
        try:
            prims = game.unreal_service.get_components_by_class(
                actor=actor, uclass="UPrimitiveComponent",
                include_from_child_actors=True,
            )
        except Exception as e:
            prims = []
            print(f"  {stable}: <prim enum failed: {e}>", flush=True)
            continue
        try:
            loc = actor.K2_GetActorLocation(as_dict=True)
            loc_s = f"({loc['x']:.0f},{loc['y']:.0f},{loc['z']:.0f})"
        except Exception:
            loc_s = "?"
        n_prims = len(prims)
        print(f"  {stable} @ {loc_s}  prim_count={n_prims}", flush=True)


def render_one(instance, cam, comp, out_name):
    import cv2

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
        with instance.begin_frame():
            destroy_pre_existing_renderables(game=game)
            spawn_point_light(
                game=game,
                x_cm=260.0, y_cm=220.0, z_cm=265.0,
                intensity_lumens=2200.0,
                attenuation_cm=600.0,
            )
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame():
            pass
        instance.step(num_frames=4)

        with instance.begin_frame():
            enumerate_all_actors_with_primitives(game=game, tag="empty scene")
        with instance.end_frame():
            pass

        render_one(instance, cam, comp, "frame_empty_scene.png")
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
