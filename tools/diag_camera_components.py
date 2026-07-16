"""Enumerate EVERY component on EVERY actor, including UActorComponent
(not just UPrimitiveComponent). And introspect the camera actor's
subhierarchy specifically. Then render one frame.
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


def _describe(handle):
    try:
        return str(handle)
    except Exception:
        return "<no-repr>"


def destroy_pre_existing(game):
    for cls in (
        "APlayerStart", "ADefaultPawn", "ASpectatorPawn",
        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
        "ADecalActor", "AInstancedFoliageActor",
        "AGameplayDebuggerCategoryReplicator",
        "AGameplayDebuggerPlayerManager",
    ):
        try:
            for a in game.unreal_service.find_actors_by_class(uclass=cls):
                try:
                    game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass
        except Exception:
            pass


def dump_all_components(game, tag):
    print(f"\n---- ALL UActorComponent tree ({tag}) ----", flush=True)
    actors = game.unreal_service.find_actors_by_class(uclass="AActor")
    for actor in actors:
        try:
            stable = game.unreal_service.get_stable_name_for_actor(
                actor=actor, include_unreal_name=True
            )
        except Exception:
            stable = "<no-stable>"
        # UActorComponent = base class of every component. Includes non-primitive.
        try:
            comps = game.unreal_service.get_components_by_class(
                actor=actor, uclass="UActorComponent",
                include_from_child_actors=True,
            )
        except Exception as e:
            print(f"  {stable}: <UActorComponent enum failed: {e}>", flush=True)
            continue
        if not comps:
            print(f"  {stable}: <no components>", flush=True)
            continue
        print(f"  {stable}: {len(comps)} components", flush=True)
        for c in comps:
            # Best-effort class name.
            cls_name = "?"
            try:
                cls_name = getattr(c, "_class_name", "?")
            except Exception:
                pass
            print(f"      - class={cls_name}  handle={_describe(c)[:120]}", flush=True)


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
            destroy_pre_existing(game=game)
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
            dump_all_components(game=game, tag="after camera spawn")
        with instance.end_frame():
            pass

        render_one(instance, cam, comp, "frame_component_dump.png")
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
