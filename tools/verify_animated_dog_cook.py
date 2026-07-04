"""Verify that Stage 2 (headless UE import + cook) produced runtime-loadable
uassets for the animated dog. Blocks Task 7 (Gate 4).

Exit 0 + 'COOK_VERIFY_OK' — proceed to T7.
Exit 1 + 'COOK_VERIFY_FAILED_<what>' — go back to T5 (re-import or re-cook).

Path names differ from the original spec (which anticipated SKM_ / SK_ / anim_
prefixes from a manual GUI import). UE 5.5 Interchange glTF import instead
keeps the source names, so the on-disk assets are:
  Dog_textured.uasset          (SkeletalMesh)
  Dog_textured_Skeleton.uasset (Skeleton)
  Walking.uasset               (AnimSequence)
  Idle.uasset                  (AnimSequence)
  Dog_Fur.uasset               (Material)
Plus BP_dog_animated.uasset in the Blueprints/ subdir.
"""

import sys

sys.path.insert(0, "/data/jzy/code/SPEAR/examples")
from render_in_gpurir_room import configure_gpurir_instance  # noqa: E402

BP_PATH = "/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C"
ANIM_WALKING_PATH = "/Game/MyAssets/Audioset/Meshes/animated_dog/Walking"
SKM_PATH = "/Game/MyAssets/Audioset/Meshes/animated_dog/Dog_textured"


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            try:
                bp = game.unreal_service.load_class(uclass="AActor", name=BP_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_class_BP: {e}", flush=True)
                sys.exit(1)
            if bp is None:
                print("COOK_VERIFY_FAILED_load_class_BP: returned None", flush=True)
                sys.exit(1)

            try:
                anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=ANIM_WALKING_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_anim_Walking: {e}", flush=True)
                sys.exit(1)
            if anim is None:
                print("COOK_VERIFY_FAILED_load_anim_Walking: returned None", flush=True)
                sys.exit(1)

            try:
                skm = game.unreal_service.load_object(uclass="USkeletalMesh", name=SKM_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_SKM: {e}", flush=True)
                sys.exit(1)
            if skm is None:
                print("COOK_VERIFY_FAILED_load_SKM: returned None", flush=True)
                sys.exit(1)

            print("COOK_VERIFY_OK", flush=True)
        with instance.end_frame():
            pass
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
