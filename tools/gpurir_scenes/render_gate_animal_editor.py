"""Headless UE editor script: import a Hunyuan textured obj as StaticMesh + BP.

Reads env vars:
  STATIC_TAG   - species tag, e.g. "horse"
  STATIC_MESH  - absolute path to textured.obj
"""
import os
import posixpath

import spear
import unreal


TAG = os.environ["STATIC_TAG"]
MESH = os.environ["STATIC_MESH"]
MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_static_{TAG}"
BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_static_{TAG}"
BP_NAME = f"BP_gate_static_{TAG}"


def _make_or_clear(path):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        spear.log(f"    clearing {path}")
        assert unreal.EditorAssetLibrary.delete_directory(directory_path=path)
    assert unreal.EditorAssetLibrary.make_directory(directory_path=path)


def main():
    assert os.path.exists(MESH), MESH
    _make_or_clear(MESH_DIR)
    _make_or_clear(BP_DIR)

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", MESH)
    task.set_editor_property("destination_path", MESH_DIR)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(import_tasks=[task])

    static_mesh_path = None
    for ap in unreal.EditorAssetLibrary.list_assets(directory_path=MESH_DIR, recursive=True):
        data = unreal.EditorAssetLibrary.find_asset_data(asset_path=ap)
        cls_name = str(data.get_editor_property("asset_class_path").get_editor_property("asset_name"))
        if cls_name == "StaticMesh":
            n = str(data.get_editor_property("asset_name"))
            pkg = str(data.get_editor_property("package_path"))
            static_mesh_path = posixpath.join(pkg, f"{n}.{n}")
            spear.log(f"  found StaticMesh: {static_mesh_path}")
            break
    assert static_mesh_path is not None, f"no StaticMesh imported into {MESH_DIR}"

    bp_path = posixpath.join(BP_DIR, BP_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        assert unreal.EditorAssetLibrary.delete_asset(asset_path_to_delete=bp_path)

    bp = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME, package_dir=BP_DIR,
        parent_class=unreal.StaticMeshActor,
    )
    subobjs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=bp)
    sm_comp = None
    for so in subobjs:
        if isinstance(so["object"], unreal.StaticMeshComponent):
            sm_comp = so["object"]
            break
    assert sm_comp is not None, "no StaticMeshComponent in default BP subobjects"
    sm_comp.set_static_mesh(new_mesh=unreal.load_asset(name=static_mesh_path))

    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(asset_to_save=bp)
    spear.log(f"IMPORT_STATIC_OK tag={TAG} sm={static_mesh_path} bp={bp_path}")


if __name__ == "__main__":
    main()
