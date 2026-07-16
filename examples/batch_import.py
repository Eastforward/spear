"""Batch-import all Hunyuan3D AudioSet GLBs into SpearSim (one editor session).
Generalizes my_import_asset.py: scans outputs/audioset_assets/*/<name>_textured.glb,
imports each -> SM_<name> + BP_<name> + bounds meta. Run via run_editor_script.
"""
import os, posixpath, json, spear, unreal

ROOT = "/data/jzy/code/Hunyuan3D-2.1/outputs/audioset_assets"
SM_BASE = "/Game/MyAssets/Audioset/Meshes"
BP_BASE = "/Game/MyAssets/Audioset/Blueprints"
META_DIR = "/data/jzy/code/SPEAR/tmp/asset_meta"
os.makedirs(META_DIR, exist_ok=True)

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
eas = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)

assets = []
for name in sorted(os.listdir(ROOT)):
    glb = os.path.join(ROOT, name, f"{name}_textured.glb")
    if os.path.exists(glb):
        assets.append((name, glb))
spear.log(f"[batch] {len(assets)} GLBs to import")


def find_static_mesh(pkg_dir):
    paths = eal.list_assets(pkg_dir, recursive=True, include_folder=False)
    for p in paths:
        obj = unreal.load_asset(p)
        if isinstance(obj, unreal.StaticMesh):
            return obj, p
    return None, None


if __name__ == "__main__":
    ok, fail = 0, 0
    for idx, (name, glb) in enumerate(assets):
        sm_dir = posixpath.join(SM_BASE, name)
        bp_dir = posixpath.join(BP_BASE, name)
        tmp_dir = f"/Game/MyAssets/Audioset/_tmp_{name}"
        sm_name = f"SM_{name}"; bp_name = f"BP_{name}"
        sm_path = posixpath.join(sm_dir, sm_name); bp_path = posixpath.join(bp_dir, bp_name)
        try:
            task = unreal.AssetImportTask()
            task.set_editor_property("async_", False); task.set_editor_property("automated", True)
            task.set_editor_property("destination_path", tmp_dir)
            task.set_editor_property("filename", glb)
            task.set_editor_property("replace_existing", True); task.set_editor_property("replace_existing_settings", True)
            task.set_editor_property("save", True)
            asset_tools.import_asset_tasks(import_tasks=[task])
            asset_obj, pkg = find_static_mesh(tmp_dir)
            if asset_obj is None:
                spear.log(f"[batch] {idx+1}/{len(assets)} {name}: NO MESH"); fail += 1; continue
            if eal.does_asset_exist(sm_path):
                eal.delete_asset(sm_path)
            rd = unreal.AssetRenameData(); rd.set_editor_property("asset", asset_obj)
            rd.set_editor_property("new_name", sm_name); rd.set_editor_property("new_package_path", sm_dir)
            asset_tools.rename_assets(assets_and_names=[rd])
            if eal.does_asset_exist(bp_path):
                eal.delete_asset(bp_path)
            bp = spear.editor.create_blueprint_asset(asset_name=bp_name, package_dir=bp_dir, parent_class=unreal.StaticMeshActor)
            descs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=bp)
            actor = descs[0]["object"]; smc = descs[1]["object"]
            sm = unreal.load_asset(sm_path)
            smc.set_editor_property("static_mesh", sm)
            bmin, bmax = smc.get_local_bounds()
            origin = (bmax + bmin) / 2.0
            actor.set_editor_property("pivot_offset", unreal.Vector(x=origin.x, y=origin.y, z=0.0))
            eas.save_loaded_asset(asset_to_save=bp)
            meta = {"ext": float(max((bmax - bmin).x, (bmax - bmin).y, (bmax - bmin).z)),
                    "bmin_z": float(bmin.z), "height": float((bmax - bmin).z)}
            json.dump(meta, open(os.path.join(META_DIR, f"{name}.json"), "w"))
            ok += 1
            if (idx + 1) % 10 == 0:
                spear.log(f"[batch] {idx+1}/{len(assets)} done ({ok} ok, {fail} fail)")
        except Exception as e:
            spear.log(f"[batch] {name}: FAIL {type(e).__name__}: {e}"); fail += 1
    spear.log(f"[batch] ALL DONE: {ok}/{len(assets)} ok, {fail} fail")
