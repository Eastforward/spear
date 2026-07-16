"""Import one reviewed stable-template humanoid GLB into an isolated UE gate.

The Hunyuan-derived appearance remains a technical spike. This script accepts
only the two reviewed Rocketbox template assets and never writes to the formal
source-asset registry.
"""

import datetime
import hashlib
import json
import os
import posixpath
import struct
from pathlib import Path

import spear
import unreal


ALLOWED_TAG_TO_ASSET_ID = {
    "hy3d_rocketbox_male_adult_01_spike": "rocketbox_male_adult_01",
    "hy3d_rocketbox_female_adult_01_spike": "rocketbox_female_adult_01",
}
USAGE_SCOPE = "technical_spike_only"
MANIFEST_SCHEMA = "hy3d_rocketbox_ue_import_v1"
EXPECTED_BONE_COUNT = 80
EXPECTED_MATERIAL_SLOTS = 3
REQUIRED_ANIMATION_NAMES = {"Walking", "Standing_Idle"}

TAG = os.environ["GATE_TAG"]
RIGGED_GLB = os.environ["GATE_RIGGED_GLB"]
IMPORT_MANIFEST = os.environ["GATE_IMPORT_MANIFEST"]
VERIFY_ONLY = os.environ.get("GATE_VERIFY_ONLY") == "1"
MESH_CONTENT_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_CONTENT_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(tmp_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _validate_environment():
    if TAG not in ALLOWED_TAG_TO_ASSET_ID:
        raise RuntimeError(f"unapproved humanoid spike tag: {TAG!r}")
    spear_root = Path(__file__).resolve().parents[1]
    asset_id = ALLOWED_TAG_TO_ASSET_ID[TAG]
    expected_glb = (
        spear_root
        / "tmp" / "hy3d_rocketbox_template_fit_v1"
        / asset_id / "ue_runtime.glb"
    ).resolve()
    source_glb = Path(RIGGED_GLB).expanduser().resolve()
    if source_glb != expected_glb:
        raise RuntimeError(f"GLB is outside reviewed spike output: {source_glb}")
    if not source_glb.is_file():
        raise RuntimeError(f"missing runtime GLB: {source_glb}")

    expected_manifest = (
        spear_root
        / "tmp" / "hy3d_rocketbox_template_fit_v1"
        / asset_id / "ue_import_manifest.json"
    ).resolve()
    manifest_path = Path(IMPORT_MANIFEST).expanduser().resolve()
    if manifest_path != expected_manifest:
        raise RuntimeError(f"manifest is outside reviewed spike output: {manifest_path}")
    return asset_id, source_glb, manifest_path


def _read_glb_contract(path):
    payload = Path(path).read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError(f"not a GLB 2.0 file: {path}")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    if version != 2 or declared_length != len(payload):
        raise RuntimeError(f"invalid GLB header: version={version} length={declared_length}")
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if chunk_type != 0x4E4F534A or 20 + json_length > len(payload):
        raise RuntimeError("GLB does not begin with a valid JSON chunk")
    document = json.loads(payload[20:20 + json_length].decode("utf-8"))
    if len(document.get("meshes", [])) != 1:
        raise RuntimeError("runtime GLB must contain exactly one mesh")
    if len(document.get("skins", [])) != 1:
        raise RuntimeError("runtime GLB must contain exactly one skin")
    skin = document["skins"][0]
    if len(skin.get("joints", [])) != EXPECTED_BONE_COUNT:
        raise RuntimeError("runtime GLB must contain the sealed 80-joint skeleton")
    animation_names = {item.get("name") for item in document.get("animations", [])}
    if animation_names != REQUIRED_ANIMATION_NAMES:
        raise RuntimeError(f"unexpected GLB animations: {sorted(animation_names)}")
    if len(document.get("materials", [])) < EXPECTED_MATERIAL_SLOTS:
        raise RuntimeError("runtime GLB is missing material-scoped PBR assets")
    return {
        "mesh_count": len(document["meshes"]),
        "skin_count": len(document["skins"]),
        "joint_count": len(skin["joints"]),
        "material_count": len(document["materials"]),
        "animation_names": sorted(animation_names),
    }


def _make_or_clear_dir(path):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        spear.log(f"Clearing existing spike directory: {path}")
        if not unreal.EditorAssetLibrary.delete_directory(directory_path=path):
            raise RuntimeError(f"could not clear UE content directory: {path}")
    if not unreal.EditorAssetLibrary.make_directory(directory_path=path):
        raise RuntimeError(f"could not create UE content directory: {path}")


def _asset_record(asset_path):
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_name = str(
        data.get_editor_property("asset_class_path").get_editor_property("asset_name")
    )
    package_path = str(data.get_editor_property("package_path"))
    asset_name = str(data.get_editor_property("asset_name"))
    object_path = posixpath.join(package_path, f"{asset_name}.{asset_name}")
    return {
        "asset_name": asset_name,
        "class_name": class_name,
        "object_path": object_path,
    }


def _collect_imported_assets():
    paths = unreal.EditorAssetLibrary.list_assets(
        directory_path=MESH_CONTENT_DIR,
        recursive=True,
        include_folder=False,
    )
    assets = {
        "skeletal_mesh": [],
        "skeleton": [],
        "animations": {},
        "materials": [],
        "textures": [],
        "other": [],
    }
    for asset_path in paths:
        record = _asset_record(asset_path)
        class_name = record["class_name"]
        if class_name == "SkeletalMesh":
            assets["skeletal_mesh"].append(record)
        elif class_name == "Skeleton":
            assets["skeleton"].append(record)
        elif class_name == "AnimSequence":
            name = record["asset_name"]
            if name in assets["animations"]:
                raise RuntimeError(f"duplicate animation asset name: {name}")
            assets["animations"][name] = record
        elif class_name in ("Material", "MaterialInstanceConstant"):
            assets["materials"].append(record)
        elif class_name == "Texture2D":
            assets["textures"].append(record)
        else:
            assets["other"].append(record)
    return assets


def _load_required(record):
    asset = unreal.load_asset(name=record["object_path"])
    if asset is None:
        raise RuntimeError(f"could not load imported asset: {record['object_path']}")
    return asset


def _create_blueprint(assets, editor_asset_subsystem):
    bp_path = posixpath.join(BP_CONTENT_DIR, BP_NAME)
    blueprint_asset = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME,
        package_dir=BP_CONTENT_DIR,
        parent_class=unreal.SkeletalMeshActor,
    )
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint_asset
    )
    skeletal_components = [
        item["object"]
        for item in subobjects
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(skeletal_components) != 1:
        raise RuntimeError(f"expected one Blueprint skeletal component, got {len(skeletal_components)}")
    smc = skeletal_components[0]
    skeletal_mesh = _load_required(assets["skeletal_mesh"][0])
    animations = {
        name: _load_required(record)
        for name, record in assets["animations"].items()
    }
    smc.set_animation_mode(animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    smc.set_skeletal_mesh_asset(new_mesh=skeletal_mesh)
    smc.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=animations["Walking"],
            saved_position=0.0,
            saved_play_rate=1.0,
        ),
    )
    smc.set_editor_property(
        name="visibility_based_anim_tick_option",
        value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )
    editor_asset_subsystem.save_loaded_asset(asset_to_save=blueprint_asset)
    return blueprint_asset, smc, bp_path


def _validate_runtime_assets(assets, smc):
    if len(assets["skeletal_mesh"]) != 1:
        raise RuntimeError(f"expected one SkeletalMesh, got {len(assets['skeletal_mesh'])}")
    if len(assets["skeleton"]) != 1:
        raise RuntimeError(f"expected one Skeleton, got {len(assets['skeleton'])}")
    if set(assets["animations"]) != set(REQUIRED_ANIMATION_NAMES):
        raise RuntimeError(f"expected exact animations, got {sorted(assets['animations'])}")
    if len(assets["materials"]) < EXPECTED_MATERIAL_SLOTS:
        raise RuntimeError(f"expected at least three materials, got {len(assets['materials'])}")
    if len(assets["textures"]) < EXPECTED_MATERIAL_SLOTS:
        raise RuntimeError(f"expected at least three textures, got {len(assets['textures'])}")

    skeletal_mesh = _load_required(assets["skeletal_mesh"][0])
    imported_skeleton = _load_required(assets["skeleton"][0])
    mesh_skeleton = skeletal_mesh.get_editor_property("skeleton")
    if mesh_skeleton is None or mesh_skeleton.get_path_name() != imported_skeleton.get_path_name():
        raise RuntimeError("SkeletalMesh does not reference the imported Skeleton")
    for record in assets["animations"].values():
        animation = _load_required(record)
        animation_skeleton = animation.get_editor_property("skeleton")
        if animation_skeleton is None or animation_skeleton.get_path_name() != imported_skeleton.get_path_name():
            raise RuntimeError(f"animation uses the wrong skeleton: {record['asset_name']}")

    bone_count = smc.get_num_bones()
    if bone_count != EXPECTED_BONE_COUNT:
        raise RuntimeError(f"expected 80 imported bones, got {bone_count}")

    material_slots = skeletal_mesh.get_editor_property("materials")
    if len(material_slots) != EXPECTED_MATERIAL_SLOTS:
        raise RuntimeError(f"expected three SkeletalMesh material slots, got {len(material_slots)}")
    slot_records = []
    for slot in material_slots:
        material_interface = slot.material_interface
        if material_interface is None:
            raise RuntimeError(f"null material interface in slot {slot.material_slot_name}")
        slot_records.append({
            "slot_name": str(slot.material_slot_name),
            "material_path": material_interface.get_path_name(),
        })

    return {"bone_count": bone_count, "material_slots": slot_records}


def _sorted_paths(records):
    return sorted(item["object_path"] for item in records)


def _build_manifest(asset_id, source_glb, glb_contract, assets, runtime, bp_path):
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": _utc_now(),
        "tag": TAG,
        "asset_id": asset_id,
        "usage_scope": USAGE_SCOPE,
        "source_glb": str(source_glb),
        "source_glb_sha256": _sha256(source_glb),
        "source_glb_size_bytes": source_glb.stat().st_size,
        "glb_contract": glb_contract,
        "content": {
            "mesh_directory": MESH_CONTENT_DIR,
            "blueprint_directory": BP_CONTENT_DIR,
            "skeletal_mesh": assets["skeletal_mesh"][0]["object_path"],
            "skeleton": assets["skeleton"][0]["object_path"],
            "animations": {
                name: assets["animations"][name]["object_path"]
                for name in sorted(REQUIRED_ANIMATION_NAMES)
            },
            "materials": _sorted_paths(assets["materials"]),
            "textures": _sorted_paths(assets["textures"]),
            "blueprint": bp_path,
        },
        "runtime_contract": runtime,
        "reload_verification": {"status": "pending"},
    }


def _cleanup_partial_import(manifest_path):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest_path.unlink()
    for content_dir in (BP_CONTENT_DIR, MESH_CONTENT_DIR):
        if unreal.EditorAssetLibrary.does_directory_exist(directory_path=content_dir):
            unreal.EditorAssetLibrary.delete_directory(directory_path=content_dir)


def _load_blueprint_component(bp_path):
    blueprint_asset = unreal.load_asset(name=bp_path)
    if blueprint_asset is None:
        raise RuntimeError(f"could not reload Blueprint: {bp_path}")
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint_asset
    )
    components = [
        item["object"]
        for item in subobjects
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(components) != 1:
        raise RuntimeError(f"reloaded Blueprint has {len(components)} skeletal components")
    return components[0]


def _verify_only(asset_id, source_glb, manifest_path):
    if not manifest_path.is_file():
        raise RuntimeError(f"missing import manifest for reload verification: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("usage_scope") != USAGE_SCOPE:
        raise RuntimeError("import manifest schema or usage scope mismatch")
    if manifest.get("tag") != TAG or manifest.get("asset_id") != asset_id:
        raise RuntimeError("import manifest identity mismatch")
    if manifest.get("source_glb_sha256") != _sha256(source_glb):
        raise RuntimeError("runtime GLB changed after import")
    assets = _collect_imported_assets()
    bp_path = manifest["content"]["blueprint"]
    smc = _load_blueprint_component(bp_path)
    runtime = _validate_runtime_assets(assets, smc)
    manifest["runtime_contract"] = runtime
    manifest["reload_verification"]["status"] = "passed"
    manifest["reload_verification"]["verified_at"] = _utc_now()
    manifest["reload_verification"]["process"] = "second_ue_commandlet"
    _atomic_write_json(manifest_path, manifest)
    spear.log(f"VERIFY_GATE_HUMANOID_OK tag={TAG} manifest={manifest_path}")


def main():
    asset_id, source_glb, manifest_path = _validate_environment()
    glb_contract = _read_glb_contract(source_glb)
    if VERIFY_ONLY:
        _verify_only(asset_id, source_glb, manifest_path)
        return

    editor_asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    try:
        if manifest_path.exists():
            manifest_path.unlink()
        _make_or_clear_dir(MESH_CONTENT_DIR)
        _make_or_clear_dir(BP_CONTENT_DIR)

        task = unreal.AssetImportTask()
        task.set_editor_property(name="async_", value=False)
        task.set_editor_property(name="automated", value=True)
        task.set_editor_property(name="destination_path", value=MESH_CONTENT_DIR)
        task.set_editor_property(name="filename", value=str(source_glb))
        task.set_editor_property(name="replace_existing", value=True)
        task.set_editor_property(name="replace_existing_settings", value=True)
        task.set_editor_property(name="save", value=True)
        asset_tools.import_asset_tasks(import_tasks=[task])

        assets = _collect_imported_assets()
        _blueprint_asset, smc, bp_path = _create_blueprint(
            assets, editor_asset_subsystem
        )
        runtime = _validate_runtime_assets(assets, smc)
        manifest = _build_manifest(
            asset_id,
            source_glb,
            glb_contract,
            assets,
            runtime,
            bp_path,
        )
        _atomic_write_json(manifest_path, manifest)
    except BaseException:
        _cleanup_partial_import(manifest_path)
        raise

    spear.log(
        f"IMPORT_GATE_HUMANOID_OK tag={TAG} "
        f"skm={assets['skeletal_mesh'][0]['object_path']} bp={bp_path} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
