"""Headless UE Editor Python script: import Dog_textured.glb + create a
Blueprint that plays the Walking anim on loop. No GUI required.

Invoked via tools/run_editor_script.py which launches UE Editor in
commandlet mode (-run=pythonscript). See docs/animated_dog_ue_import.md
for the equivalent manual GUI steps. This script IS the automation of
those steps.

Cross-refs:
- spec 2026-07-04-animated-dog-gpurir-design.md §Component B
- plan 2026-07-04-animated-dog-gpurir.md §Task 5
- reference impl: examples/import_mixamo_dataset/import_dataset.py
"""

import argparse
import posixpath
import sys

import spear
import unreal


DEFAULT_GLB = "/data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb"
DEFAULT_MESH_DIR = "/Game/MyAssets/Audioset/Meshes/animated_dog"
DEFAULT_BP_DIR = "/Game/MyAssets/Audioset/Blueprints/animated_dog"
BP_NAME = "BP_dog_animated"
WALKING_ANIM_HINT = "Walking"  # sub-string used to identify the walk AnimSequence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glb-path", default=DEFAULT_GLB)
    p.add_argument("--mesh-content-dir", default=DEFAULT_MESH_DIR)
    p.add_argument("--bp-content-dir", default=DEFAULT_BP_DIR)
    p.add_argument("--bp-name", default=BP_NAME)
    p.add_argument("--walking-hint", default=WALKING_ANIM_HINT,
                   help="Sub-string that identifies the Walking AnimSequence")
    return p.parse_args()


def _make_or_clear_dir(path):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        spear.log(f"    Directory exists, clearing: {path}")
        assert unreal.EditorAssetLibrary.delete_directory(directory_path=path)
    assert unreal.EditorAssetLibrary.make_directory(directory_path=path)


def main():
    args = parse_args()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    editor_asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)

    spear.log(f"Importing glb: {args.glb_path}")
    spear.log(f"  -> mesh content dir: {args.mesh_content_dir}")
    spear.log(f"  -> BP content dir:   {args.bp_content_dir}")

    # 1. Prep destination directories (clean re-import each time)
    _make_or_clear_dir(args.mesh_content_dir)
    _make_or_clear_dir(args.bp_content_dir)

    # 2. Import the glb
    task = unreal.AssetImportTask()
    task.set_editor_property(name="async_", value=False)
    task.set_editor_property(name="automated", value=True)
    task.set_editor_property(name="destination_path", value=args.mesh_content_dir)
    task.set_editor_property(name="filename", value=args.glb_path)
    task.set_editor_property(name="replace_existing", value=True)
    task.set_editor_property(name="replace_existing_settings", value=True)
    task.set_editor_property(name="save", value=True)
    asset_tools.import_asset_tasks(import_tasks=[task])
    assert unreal.EditorAssetLibrary.does_directory_exist(directory_path=args.mesh_content_dir)

    imported_paths = unreal.EditorAssetLibrary.list_assets(directory_path=args.mesh_content_dir)
    spear.log("Imported assets:")
    for ap in imported_paths:
        spear.log(f"  {ap}")

    # 3. Locate the SkeletalMesh and the Walking AnimSequence among imported assets
    skeletal_mesh_path = None
    walking_anim_path = None
    for ap in imported_paths:
        data = unreal.EditorAssetLibrary.find_asset_data(asset_path=ap)
        cls = str(data.get_editor_property(name="asset_class_path").get_editor_property(name="asset_name"))
        pkg_dir = str(data.get_editor_property(name="package_path"))
        name = str(data.get_editor_property(name="asset_name"))
        obj_path = posixpath.join(pkg_dir, f"{name}.{name}")
        if cls == "SkeletalMesh":
            skeletal_mesh_path = obj_path
            spear.log(f"  found SkeletalMesh: {obj_path}")
        elif cls == "AnimSequence" and args.walking_hint.lower() in name.lower():
            walking_anim_path = obj_path
            spear.log(f"  found Walking AnimSequence: {obj_path}")

    assert skeletal_mesh_path is not None, f"no SkeletalMesh found in {imported_paths}"
    assert walking_anim_path is not None, (
        f"no AnimSequence with '{args.walking_hint}' in name found among {imported_paths}"
    )

    # 4. Create the Blueprint wrapping the SkeletalMesh with the Walking anim on loop
    bp_path = posixpath.join(args.bp_content_dir, args.bp_name)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        spear.log(f"BP exists, removing: {bp_path}")
        assert unreal.EditorAssetLibrary.delete_asset(asset_path_to_delete=bp_path)

    spear.log(f"Creating BP: {bp_path}")
    blueprint_asset = spear.editor.create_blueprint_asset(
        asset_name=args.bp_name,
        package_dir=args.bp_content_dir,
        parent_class=unreal.SkeletalMeshActor,
    )
    subobjs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=blueprint_asset)
    assert len(subobjs) == 2
    assert isinstance(subobjs[0]["object"], unreal.SkeletalMeshActor)
    assert isinstance(subobjs[1]["object"], unreal.SkeletalMeshComponent)

    smc = subobjs[1]["object"]
    smc.set_animation_mode(animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    smc.set_skeletal_mesh_asset(new_mesh=unreal.load_asset(name=skeletal_mesh_path))
    anim = unreal.load_asset(name=walking_anim_path)
    # animation_data is a SingleAnimationPlayData: anim_to_play + looping + playing
    # (fields discovered via `help(unreal.SingleAnimationPlayData)` — bare
    # 'playing'/'looping' as SMC properties don't exist on this UE version).
    smc.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=anim,
            saved_position=0.0,
            saved_play_rate=1.0,
        ),
    )

    spear.log(f"Saving BP: {bp_path}")
    editor_asset_subsystem.save_loaded_asset(asset_to_save=blueprint_asset)

    # 5. Print a summary line the caller can grep for
    spear.log(f"IMPORT_ANIMATED_DOG_OK skm={skeletal_mesh_path} anim={walking_anim_path} bp={bp_path}")


if __name__ == "__main__":
    main()
