"""Probe: does SPEAR RPC actually let us drive USkeletalMeshComponent
PlayAnimation? Spec 2026-07-04-animated-dog-gpurir §F5 / Gate 0.

Uses SKM_Manny_Simple + MF_Walk_Fwd (both already cooked in SpearSim) so we
don't need our own animated dog asset yet. Exit 0 + print 'PROBE_OK' => proceed
with Task 3/7 as designed. Exit 1 + print 'PROBE_FAILED_<reason>' => trigger F5
fallback (define a BlueprintCallable helper inside BP_dog_animated and call
that instead of raw PlayAnimation).
"""

import os
import sys

sys.path.insert(0, "/data/jzy/code/SPEAR/examples")
from render_in_gpurir_room import configure_gpurir_instance  # noqa: E402

MANNEQUIN_MESH = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple.SKM_Manny_Simple"
WALK_ANIM = "/Game/Characters/Mannequins/Animations/Quinn/MF_Walk_Fwd.MF_Walk_Fwd"


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    probe_ok = False
    try:
        try:
            with instance.begin_frame():
                # Kill spawn cube etc. so the probe is clean
                for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn"):
                    try:
                        for a in game.unreal_service.find_actors_by_class(uclass=cls):
                            game.unreal_service.destroy_actor(actor=a)
                    except Exception:
                        pass

                # Spawn a bare AActor and add a USkeletalMeshComponent by hand.
                # Simpler than fishing a BP; proves the raw component-level RPC path.
                actor = game.unreal_service.spawn_actor(
                    uclass="AActor",
                    location={"X": 100.0, "Y": 100.0, "Z": 50.0},
                    spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
                )
                # AActor doesn't auto-create a SkeletalMeshComponent — many SPEAR
                # workflows spawn a BP that owns one. We instead use the character BP
                # path (proven-working in examples/control_character/run.py) so the
                # probe measures the same call sequence T7 will use later.
                bp_uclass = game.unreal_service.load_class(
                    uclass="AActor",
                    name="/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter.BP_ThirdPersonCharacter_C",
                )
                game.unreal_service.destroy_actor(actor=actor)  # drop the bare actor
                actor = game.unreal_service.spawn_actor(
                    uclass=bp_uclass,
                    location={"X": 100.0, "Y": 100.0, "Z": 100.0},
                    spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
                )
                smc = game.unreal_service.get_component_by_class(
                    actor=actor, uclass="USkeletalMeshComponent"
                )
                # Load + assign the mannequin mesh so PlayAnimation has a skeleton to drive
                mesh = game.unreal_service.load_object(uclass="USkeletalMesh", name=MANNEQUIN_MESH)
                smc.SetSkeletalMeshAsset(NewMesh=mesh)

                # THE PROBE — single PlayAnimation UFUNCTION call. UE 5.5
                # native param names are `NewAnimToPlay` (UAnimationAsset*)
                # and `bLooping` (bool); see SkeletalMeshComponent.h:1126.
                # This is preferable to the SetAnimationMode+SetAnimation+Play
                # trio because SetAnimationMode's actual UFUNCTION param is
                # `InAnimationMode`, and passing a wrong kwarg silently drives
                # the SPEAR RPC layer into an error state (the C-side asserts
                # and returns default null on subsequent calls, so a naive
                # `except Exception` won't see it — the earlier probe printed
                # PROBE_OK while the server was in a persistent error state).
                anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=WALK_ANIM)
                if anim is None:
                    print("PROBE_FAILED_load_anim_asset returned None", flush=True)
                    sys.exit(1)

                try:
                    smc.PlayAnimation(NewAnimToPlay=anim, bLooping=True)
                except Exception as e:
                    print(f"PROBE_FAILED_PlayAnimation: {e}", flush=True)
                    sys.exit(1)

                # All probe steps succeeded
                probe_ok = True

        except Exception:
            # If anything fails, frame exit exception is caught here
            pass

        # Try to exit frame gracefully
        try:
            with instance.end_frame():
                pass
        except Exception:
            # Frame exit may fail due to RPC service state
            pass

        # Print PROBE_OK only if all RPC calls succeeded
        if probe_ok:
            print("PROBE_OK", flush=True)

    finally:
        try:
            instance.close(force=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
