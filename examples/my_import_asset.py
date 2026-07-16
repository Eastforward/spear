"""Pilot: import ONE Hunyuan3D .glb into SpearSim as StaticMesh + Blueprint.
Runs INSIDE the UE editor via run_editor_script.py. Find mesh via list_assets (not ARFilter).
"""
import os, posixpath, spear, unreal

GLTF_FILE = "/data/jzy/code/Hunyuan3D-2.1/outputs/audioset_assets/clock/clock_textured.glb"
ASSET_NAME = "Clock"
SM_DIR = "/Game/MyAssets/Meshes"
BP_DIR = "/Game/MyAssets/Blueprints"
TMP_DIR = f"/Game/MyAssets/_tmp_{ASSET_NAME}"

sm_name = f"SM_{ASSET_NAME}"; bp_name = f"BP_{ASSET_NAME}"
sm_path = posixpath.join(SM_DIR, sm_name); bp_path = posixpath.join(BP_DIR, bp_name)

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary


def find_static_mesh_in(pkg_dir):
    """Return (loaded StaticMesh, package_path) for the first StaticMesh under pkg_dir."""
    paths = eal.list_assets(pkg_dir, recursive=True, include_folder=False)
    for p in paths:
        obj = unreal.load_asset(p)
        if isinstance(obj, unreal.StaticMesh):
            return obj, p
    return None, None


if __name__ == "__main__":
    spear.log(f"[myimport] importing {GLTF_FILE}")

    # 1) import the glb into a per-asset temp dir
    task = unreal.AssetImportTask()
    task.set_editor_property("async_", False)
    task.set_editor_property("automated", True)
    task.set_editor_property("destination_path", TMP_DIR)
    task.set_editor_property("filename", GLTF_FILE)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("replace_existing_settings", True)
    task.set_editor_property("save", True)
    asset_tools.import_asset_tasks(import_tasks=[task])
    spear.log("[myimport] import task done")

    # 2) find the imported StaticMesh
    asset, imported_pkg = find_static_mesh_in(TMP_DIR)
    spear.log(f"[myimport] found StaticMesh: {imported_pkg}")
    assert asset is not None, "no StaticMesh imported"

    # 3) move/rename to SM_DIR/SM_<name>
    if eal.does_asset_exist(sm_path):
        eal.delete_asset(sm_path)
    rd = unreal.AssetRenameData()
    rd.set_editor_property("asset", asset)
    rd.set_editor_property("new_name", sm_name)
    rd.set_editor_property("new_package_path", SM_DIR)
    asset_tools.rename_assets(assets_and_names=[rd])
    spear.log(f"[myimport] renamed -> {sm_path}")

    # 4) create BP_<name>
    if eal.does_asset_exist(bp_path):
        eal.delete_asset(bp_path)
    bp = spear.editor.create_blueprint_asset(asset_name=bp_name, package_dir=BP_DIR, parent_class=unreal.StaticMeshActor)
    descs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=bp)
    actor = descs[0]["object"]; smc = descs[1]["object"]
    sm = unreal.load_asset(sm_path)
    smc.set_editor_property("static_mesh", sm)
    bmin, bmax = smc.get_local_bounds()
    origin = (bmax + bmin) / 2.0; half = (bmax - bmin) / 2.0
    actor.set_editor_property("pivot_offset", unreal.Vector(x=origin.x, y=origin.y, z=0.0))  # xy-center only; z grounded at render via -bmin_z*scale (no double-counting)
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(asset_to_save=bp)
    # write bounds metadata for the render script (game-session proxy can't call get_local_bounds)
    import json as _json
    meta = {"ext": float(max((bmax - bmin).x, (bmax - bmin).y, (bmax - bmin).z)),
            "bmin_z": float(bmin.z), "height": float((bmax - bmin).z)}
    _meta_dir = "/data/jzy/code/SPEAR/tmp/asset_meta"; os.makedirs(_meta_dir, exist_ok=True)
    _json.dump(meta, open(f"{_meta_dir}/{ASSET_NAME}.json", "w"))
    spear.log(f"[myimport] DONE -> BP {bp_path}, SM {sm_path}, meta={meta}")

    # 5) quit the editor (run_editor_script doesn't auto-quit; avoid idle-to-timeout)
    spear.log("[myimport] requesting editor quit")
    unreal.SystemLibrary.quit_editor()
