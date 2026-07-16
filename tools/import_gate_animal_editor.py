"""Headless UE Editor Python script: import a per-tag rigged GLB (from
robust_skin_transfer) and wrap it in a Blueprint that plays the Walking
anim on loop.

Structurally a tag-parameterized clone of import_animated_dog_editor.py.
Reads env vars:
  GATE_TAG          - species tag, e.g. "cat_persian"
  GATE_RIGGED_GLB   - absolute path to the rigged skeletal-mesh GLB

Emits Blueprint at /Game/MyAssets/Audioset/Blueprints/gate_{TAG}/BP_gate_{TAG}.
"""

import os
import posixpath

import spear
import unreal


TAG = os.environ["GATE_TAG"]
RIGGED_GLB = os.environ["GATE_RIGGED_GLB"]
MESH_CONTENT_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_CONTENT_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"
WALKING_ANIM_HINT = "walk"  # matches both "Walk" (farm rigs) and "Walking" (animalpack)
FUR_MAT_HINT = "fur"  # blender_robust_swap_mesh_keep_rig names its material "RobustSwap_Fur"


def _make_or_clear_dir(path):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        spear.log(f"    Directory exists, clearing: {path}")
        assert unreal.EditorAssetLibrary.delete_directory(directory_path=path)
    assert unreal.EditorAssetLibrary.make_directory(directory_path=path)


def main():
    assert os.path.exists(RIGGED_GLB), f"missing rigged glb: {RIGGED_GLB}"
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    editor_asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)

    spear.log(f"Importing glb: {RIGGED_GLB}")
    spear.log(f"  -> mesh content dir: {MESH_CONTENT_DIR}")
    spear.log(f"  -> BP content dir:   {BP_CONTENT_DIR}")

    _make_or_clear_dir(MESH_CONTENT_DIR)
    _make_or_clear_dir(BP_CONTENT_DIR)

    task = unreal.AssetImportTask()
    task.set_editor_property(name="async_", value=False)
    task.set_editor_property(name="automated", value=True)
    task.set_editor_property(name="destination_path", value=MESH_CONTENT_DIR)
    task.set_editor_property(name="filename", value=RIGGED_GLB)
    task.set_editor_property(name="replace_existing", value=True)
    task.set_editor_property(name="replace_existing_settings", value=True)
    task.set_editor_property(name="save", value=True)
    asset_tools.import_asset_tasks(import_tasks=[task])
    assert unreal.EditorAssetLibrary.does_directory_exist(directory_path=MESH_CONTENT_DIR)

    imported_paths = unreal.EditorAssetLibrary.list_assets(directory_path=MESH_CONTENT_DIR)
    spear.log("Imported assets:")
    for ap in imported_paths:
        spear.log(f"  {ap}")

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
        elif cls == "AnimSequence" and WALKING_ANIM_HINT.lower() in name.lower():
            walking_anim_path = obj_path
            spear.log(f"  found Walking AnimSequence: {obj_path}")
        elif cls in ("Material", "MaterialInstanceConstant") and FUR_MAT_HINT in name.lower():
            fur_material_path = obj_path
            spear.log(f"  found fur material: {obj_path} (cls={cls})")

    assert skeletal_mesh_path is not None, f"no SkeletalMesh found in {imported_paths}"
    assert walking_anim_path is not None, (
        f"no AnimSequence with '{WALKING_ANIM_HINT}' in name found among {imported_paths}"
    )

    # Bind the fur material into slot 0 so the textured mesh survives cook.
    if fur_material_path is not None:
        skm_obj = unreal.load_asset(name=skeletal_mesh_path)
        fur_mat = unreal.load_asset(name=fur_material_path)
        materials = skm_obj.get_editor_property("materials")
        if len(materials) > 0:
            materials[0].material_interface = fur_mat
            skm_obj.set_editor_property("materials", materials)
            editor_asset_subsystem.save_loaded_asset(asset_to_save=skm_obj)
            spear.log(f"  bound {fur_material_path} to SKM slot 0 and saved")

        # Ensure baseColorTexture points at the correct Texture2D and clear
        # the baseColorFactor to white so the raw Hunyuan diffuse survives.
        diffuse_path = None
        for ap in imported_paths:
            data = unreal.EditorAssetLibrary.find_asset_data(asset_path=ap)
            cls = str(data.get_editor_property(name="asset_class_path").get_editor_property(name="asset_name"))
            name = str(data.get_editor_property(name="asset_name"))
            pkg_dir = str(data.get_editor_property(name="package_path"))
            if cls == "Texture2D":
                diffuse_path = posixpath.join(pkg_dir, f"{name}.{name}")
                spear.log(f"  candidate Texture2D: {diffuse_path}")
                break
        if diffuse_path is not None:
            tex = unreal.load_asset(name=diffuse_path)
            for pname in ("baseColorTexture", "BaseColor", "DiffuseColor"):
                try:
                    ok = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                        instance=fur_mat, parameter_name=pname, value=tex,
                    )
                    spear.log(f"  set param {pname!r} -> {diffuse_path}: {ok}")
                    if ok:
                        break
                except Exception as e:
                    spear.log(f"    set param {pname!r} failed: {e}")
            # White tint so the atlas colours pass through unmodified.
            tint = unreal.LinearColor(1.0, 1.0, 1.0, 1.0)
            for pname in ("baseColorFactor", "BaseColorFactor"):
                try:
                    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
                        instance=fur_mat, parameter_name=pname, value=tint,
                    )
                    spear.log(f"  set factor {pname!r} -> white passthrough")
                except Exception:
                    pass
            for pname, val in (("metallicFactor", 0.0), ("MetallicFactor", 0.0),
                               ("roughnessFactor", 0.95), ("RoughnessFactor", 0.95)):
                try:
                    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
                        instance=fur_mat, parameter_name=pname, value=val,
                    )
                except Exception:
                    pass
            editor_asset_subsystem.save_loaded_asset(asset_to_save=fur_mat)

    bp_path = posixpath.join(BP_CONTENT_DIR, BP_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        spear.log(f"BP exists, removing: {bp_path}")
        assert unreal.EditorAssetLibrary.delete_asset(asset_path_to_delete=bp_path)

    spear.log(f"Creating BP: {bp_path}")
    blueprint_asset = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME,
        package_dir=BP_CONTENT_DIR,
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
    smc.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=anim,
            saved_position=0.0,
            saved_play_rate=1.0,
        ),
    )
    smc.set_editor_property(
        name="visibility_based_anim_tick_option",
        value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )

    spear.log(f"Saving BP: {bp_path}")
    editor_asset_subsystem.save_loaded_asset(asset_to_save=blueprint_asset)

    spear.log(f"IMPORT_GATE_ANIMAL_OK tag={TAG} skm={skeletal_mesh_path} anim={walking_anim_path} bp={bp_path}")


if __name__ == "__main__":
    main()
