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

    # 3. Locate the SkeletalMesh, Walking AnimSequence, and fur Material
    skeletal_mesh_path = None
    walking_anim_path = None
    fur_material_path = None
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
        elif cls in ("Material", "MaterialInstanceConstant") and "fur" in name.lower():
            fur_material_path = obj_path
            spear.log(f"  found fur material: {obj_path} (cls={cls})")

    assert skeletal_mesh_path is not None, f"no SkeletalMesh found in {imported_paths}"
    assert walking_anim_path is not None, (
        f"no AnimSequence with '{args.walking_hint}' in name found among {imported_paths}"
    )
    if fur_material_path is None:
        spear.log("WARN no 'fur' material found; SKM will use the Interchange default which may be pale beige")

    # 3b. Force-bind the fur material into the SKM's material slot 0 so the
    # cooked mesh renders with our procedural fur. Interchange sometimes
    # imports the Material asset but leaves the SKM slot bound to a placeholder;
    # this reassignment guarantees the correct diffuse survives cook + package.
    if fur_material_path is not None:
        skm_obj = unreal.load_asset(name=skeletal_mesh_path)
        fur_mat = unreal.load_asset(name=fur_material_path)
        materials = skm_obj.get_editor_property("materials")
        spear.log(f"  SKM material slots before: {[str(m.material_slot_name) + '=' + (m.material_interface.get_path_name() if m.material_interface else 'None') for m in materials]}")
        if len(materials) > 0:
            materials[0].material_interface = fur_mat
            skm_obj.set_editor_property("materials", materials)
            editor_asset_subsystem.save_loaded_asset(asset_to_save=skm_obj)
            spear.log(f"  bound {fur_material_path} to SKM slot 0 and saved")
        else:
            spear.log("WARN SKM has 0 material slots; can't bind fur material")

        # Diagnose the fur MaterialInstance: dump its texture parameters so we
        # can confirm the base-color texture actually points at
        # `dog_fur_diffuse` (the diffuse baked from Blender). Interchange
        # sometimes creates the material but wires the color to a placeholder.
        try:
            for pname in ["baseColorTexture", "BaseColor", "DiffuseColor", "diffuse", "Base Color"]:
                try:
                    tv = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(
                        instance=fur_mat, parameter_name=pname
                    )
                    if tv is not None:
                        spear.log(f"    fur mat texture param {pname!r} -> {tv.get_path_name()}")
                except Exception:
                    pass
            for pname in ["baseColorFactor", "BaseColorFactor", "TintColor"]:
                try:
                    vv = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(
                        instance=fur_mat, parameter_name=pname
                    )
                    spear.log(f"    fur mat vector param {pname!r} -> {vv}")
                except Exception:
                    pass
        except Exception as e:
            spear.log(f"    (couldn't dump material params: {e})")

        # Force the diffuse texture into the fur MaterialInstance's base color
        # texture parameter. The Interchange glTF importer names it
        # "baseColorTexture" for MetallicRoughness workflow.
        diffuse_path = None
        for ap in imported_paths:
            data = unreal.EditorAssetLibrary.find_asset_data(asset_path=ap)
            cls = str(data.get_editor_property(name="asset_class_path").get_editor_property(name="asset_name"))
            name = str(data.get_editor_property(name="asset_name"))
            pkg_dir = str(data.get_editor_property(name="package_path"))
            if cls == "Texture2D" and "fur" in name.lower():
                diffuse_path = posixpath.join(pkg_dir, f"{name}.{name}")
                spear.log(f"  found fur diffuse Texture2D: {diffuse_path}")
                break
        if diffuse_path is not None:
            tex = unreal.load_asset(name=diffuse_path)
            # Verify texture properties: srgb should be True for base color,
            # mip gen should be default, compression should NOT be uncompressed.
            spear.log(f"  fur diffuse Texture2D:")
            spear.log(f"    srgb={tex.get_editor_property('srgb')}")
            spear.log(f"    compression_settings={tex.get_editor_property('compression_settings')}")
            spear.log(f"    mip_gen_settings={tex.get_editor_property('mip_gen_settings')}")
            spear.log(f"    lod_group={tex.get_editor_property('lod_group')}")
            # dump parent material and its parent chain
            try:
                parent = fur_mat.get_editor_property("parent")
                spear.log(f"  fur mat parent = {parent.get_path_name() if parent else 'None'}")
            except Exception as e:
                spear.log(f"    (couldn't dump fur mat parent: {e})")
            for pname in ["baseColorTexture", "BaseColor", "DiffuseColor"]:
                try:
                    ok = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                        instance=fur_mat, parameter_name=pname, value=tex
                    )
                    spear.log(f"  set fur mat texture param {pname!r} -> {diffuse_path}: {ok}")
                    if ok:
                        break
                except Exception as e:
                    spear.log(f"    set param {pname!r} failed: {e}")
            # Force dark warm brown via baseColorFactor. Our procedural
            # diffuse averages to (156,132,106) RGB which is much too pale
            # once ambient light is factored in. Multiplying by a rich brown
            # tint pulls it back to a real dog-fur tone.
            for pname in ("baseColorFactor", "BaseColorFactor"):
                try:
                    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
                        instance=fur_mat, parameter_name=pname,
                        value=unreal.LinearColor(0.35, 0.20, 0.11, 1.0),
                    )
                    spear.log(f"  set fur mat vector param {pname!r} -> warm brown tint")
                except Exception as e:
                    spear.log(f"    set factor {pname!r} failed: {e}")
            # Turn off metallic/roughness sheen if present so the fur reads matte.
            for pname, val in (("metallicFactor", 0.0), ("MetallicFactor", 0.0),
                                ("roughnessFactor", 0.95), ("RoughnessFactor", 0.95)):
                try:
                    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
                        instance=fur_mat, parameter_name=pname, value=val,
                    )
                    spear.log(f"  set fur mat scalar param {pname!r} -> {val}")
                except Exception:
                    pass
            editor_asset_subsystem.save_loaded_asset(asset_to_save=fur_mat)

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
    # CRITICAL for cooked-build rendering via USceneCaptureComponent2D:
    # SkeletalMeshComponent's default `VisibilityBasedAnimTickOption` is
    # `OnlyTickPoseWhenRendered`, which checks `LastRenderTime`. A scene-
    # capture component (which is how SPEAR grabs frames) does NOT update
    # `LastRenderTime` reliably before the anim-tick pre-pass runs, so the
    # anim clock never advances → legs freeze in first-frame pose.
    # `AlwaysTickPoseAndRefreshBones` bypasses the visibility check entirely.
    # Setting this at the editor level bakes it into the BP's SMC default
    # subobject; the runtime `SetVisibilityBasedAnimTickOption(...)` setter
    # is not a UFUNCTION (SPEAR RPC returns 'UnrealObject' not callable).
    smc.set_editor_property(
        name="visibility_based_anim_tick_option",
        value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )
    # NB: don't try `component_tick_enabled` here — that property is on
    # `FTickFunction` (`smc.primary_component_tick.b_start_with_tick_enabled`),
    # not a bare SMC property. Component tick is already enabled by default.

    spear.log(f"Saving BP: {bp_path}")
    editor_asset_subsystem.save_loaded_asset(asset_to_save=blueprint_asset)

    # 5. Print a summary line the caller can grep for
    spear.log(f"IMPORT_ANIMATED_DOG_OK skm={skeletal_mesh_path} anim={walking_anim_path} bp={bp_path}")


if __name__ == "__main__":
    main()
