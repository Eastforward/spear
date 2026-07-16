"""Import the approved Route-2 TokenRig Walk/Idle pair into isolated UE content.

This is deliberately a research-candidate fast lane.  It never writes the
formal source-asset registry and never replaces an existing UE directory.
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


TAG = "route2_tokenrig_male_fastlane_v1"
ASSET_ID = "rocketbox_male_adult_01"
SCHEMA = "route2_tokenrig_ue_fastlane_import_v1"
USAGE_SCOPE = "research_candidate_fastlane"
EXPECTED_JOINTS = 52
REQUIRED_ANIMATIONS = {"Walking", "Standing_Idle"}

SOURCE_GLB = Path(os.environ["ROUTE2_FASTLANE_GLB"]).resolve()
MANIFEST_PATH = Path(os.environ["ROUTE2_FASTLANE_MANIFEST"]).resolve()
VERIFY_ONLY = os.environ.get("ROUTE2_FASTLANE_VERIFY_ONLY") == "1"

MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_inputs():
    root = Path(__file__).resolve().parents[1]
    expected_root = (
        root / "tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01"
    ).resolve()
    expected_glb = expected_root / "walk_idle_ue_png.glb"
    expected_manifest = expected_root / "ue_import_manifest.json"
    if SOURCE_GLB != expected_glb or not SOURCE_GLB.is_file():
        raise RuntimeError(f"unexpected fast-lane GLB: {SOURCE_GLB}")
    if MANIFEST_PATH != expected_manifest:
        raise RuntimeError(f"unexpected fast-lane manifest path: {MANIFEST_PATH}")
    transcode_path = expected_root / "ue_texture_transcode_manifest.json"
    if not transcode_path.is_file():
        raise RuntimeError("missing authenticated WebP-to-PNG transcode evidence")
    transcode = json.loads(transcode_path.read_text(encoding="utf-8"))
    if (
        transcode.get("schema") != "glb_embedded_webp_to_png_transcode_v1"
        or Path(transcode.get("output", {}).get("path", "")).resolve() != SOURCE_GLB
        or transcode.get("output", {}).get("sha256") != _sha256(SOURCE_GLB)
        or transcode.get("geometry_skin_animation_byte_graph_changed") is not False
    ):
        raise RuntimeError("WebP-to-PNG transcode evidence changed")
    return root


def _read_glb_contract(path):
    payload = Path(path).read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("fast-lane input is not GLB 2.0")
    version, declared = struct.unpack_from("<II", payload, 4)
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if version != 2 or declared != len(payload) or chunk_type != 0x4E4F534A:
        raise RuntimeError("fast-lane GLB header is invalid")
    document = json.loads(payload[20 : 20 + json_length].decode("utf-8"))
    if len(document.get("meshes", [])) != 1 or len(document.get("skins", [])) != 1:
        raise RuntimeError("fast-lane GLB must contain one mesh and one skin")
    joints = document["skins"][0].get("joints", [])
    animations = {item.get("name") for item in document.get("animations", [])}
    if len(joints) != EXPECTED_JOINTS:
        raise RuntimeError(f"expected {EXPECTED_JOINTS} TokenRig joints, got {len(joints)}")
    if animations != REQUIRED_ANIMATIONS:
        raise RuntimeError(f"unexpected actions: {sorted(animations)}")
    if not document.get("materials") or len(document.get("images", [])) < 2:
        raise RuntimeError("fast-lane GLB lost packed PBR assets")
    return {
        "mesh_count": 1,
        "skin_count": 1,
        "joint_count": len(joints),
        "animation_names": sorted(animations),
        "material_count": len(document.get("materials", [])),
        "texture_count": len(document.get("textures", [])),
        "image_count": len(document.get("images", [])),
    }


def _asset_record(asset_path):
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_name = str(
        data.get_editor_property("asset_class_path").get_editor_property("asset_name")
    )
    package_path = str(data.get_editor_property("package_path"))
    name = str(data.get_editor_property("asset_name"))
    return {
        "asset_name": name,
        "class_name": class_name,
        "object_path": posixpath.join(package_path, f"{name}.{name}"),
    }


def _collect_assets():
    assets = {
        "skeletal_mesh": [],
        "skeleton": [],
        "animations": {},
        "materials": [],
        "textures": [],
        "other": [],
    }
    paths = unreal.EditorAssetLibrary.list_assets(
        directory_path=MESH_DIR, recursive=True, include_folder=False
    )
    for path in paths:
        record = _asset_record(path)
        kind = record["class_name"]
        if kind == "SkeletalMesh":
            assets["skeletal_mesh"].append(record)
        elif kind == "Skeleton":
            assets["skeleton"].append(record)
        elif kind == "AnimSequence":
            name = record["asset_name"]
            if name in assets["animations"]:
                raise RuntimeError(f"duplicate animation: {name}")
            assets["animations"][name] = record
        elif kind in ("Material", "MaterialInstanceConstant"):
            assets["materials"].append(record)
        elif kind == "Texture2D":
            assets["textures"].append(record)
        else:
            assets["other"].append(record)
    return assets


def _load(record):
    value = unreal.load_asset(name=record["object_path"])
    if value is None:
        raise RuntimeError(f"could not load UE asset: {record['object_path']}")
    return value


def _validate_assets(assets, component):
    if len(assets["skeletal_mesh"]) != 1 or len(assets["skeleton"]) != 1:
        raise RuntimeError("UE import did not create exactly one mesh and skeleton")
    if set(assets["animations"]) != REQUIRED_ANIMATIONS:
        raise RuntimeError(f"UE import actions differ: {sorted(assets['animations'])}")
    if not assets["materials"] or len(assets["textures"]) < 2:
        raise RuntimeError("UE import lost PBR material or textures")
    mesh = _load(assets["skeletal_mesh"][0])
    skeleton = _load(assets["skeleton"][0])
    if mesh.get_editor_property("skeleton").get_path_name() != skeleton.get_path_name():
        raise RuntimeError("UE skeletal mesh references the wrong skeleton")
    for record in assets["animations"].values():
        if _load(record).get_editor_property("skeleton").get_path_name() != skeleton.get_path_name():
            raise RuntimeError(f"animation references wrong skeleton: {record['asset_name']}")
    bone_count = component.get_num_bones()
    if bone_count != EXPECTED_JOINTS:
        raise RuntimeError(f"UE imported {bone_count} bones, expected {EXPECTED_JOINTS}")
    slots = mesh.get_editor_property("materials")
    if not slots or any(slot.material_interface is None for slot in slots):
        raise RuntimeError("UE skeletal mesh material slot is empty")
    return {
        "bone_count": bone_count,
        "material_slots": [
            {
                "slot_name": str(slot.material_slot_name),
                "material_path": slot.material_interface.get_path_name(),
            }
            for slot in slots
        ],
    }


def _create_blueprint(assets):
    bp_path = posixpath.join(BP_DIR, BP_NAME)
    blueprint = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME,
        package_dir=BP_DIR,
        parent_class=unreal.SkeletalMeshActor,
    )
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint
    )
    components = [
        item["object"]
        for item in subobjects
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(components) != 1:
        raise RuntimeError("fast-lane Blueprint lacks one skeletal component")
    component = components[0]
    component.set_animation_mode(animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    component.set_skeletal_mesh_asset(new_mesh=_load(assets["skeletal_mesh"][0]))
    component.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=_load(assets["animations"]["Walking"]),
            saved_position=0.0,
            saved_play_rate=1.0,
        ),
    )
    component.set_editor_property(
        name="visibility_based_anim_tick_option",
        value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(
        asset_to_save=blueprint
    )
    return bp_path, component


def _content(assets, bp_path):
    return {
        "mesh_directory": MESH_DIR,
        "blueprint_directory": BP_DIR,
        "skeletal_mesh": assets["skeletal_mesh"][0]["object_path"],
        "skeleton": assets["skeleton"][0]["object_path"],
        "animations": {
            name: assets["animations"][name]["object_path"]
            for name in sorted(REQUIRED_ANIMATIONS)
        },
        "materials": sorted(item["object_path"] for item in assets["materials"]),
        "textures": sorted(item["object_path"] for item in assets["textures"]),
        "blueprint": bp_path,
    }


def _cleanup():
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    for path in (BP_DIR, MESH_DIR):
        if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
            unreal.EditorAssetLibrary.delete_directory(directory_path=path)


def _verify_existing(glb_contract):
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("missing fast-lane UE import manifest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("usage_scope") != USAGE_SCOPE
        or manifest.get("source_glb_sha256") != _sha256(SOURCE_GLB)
    ):
        raise RuntimeError("fast-lane UE manifest changed")
    assets = _collect_assets()
    blueprint = unreal.load_asset(name=manifest["content"]["blueprint"])
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint
    )
    components = [
        item["object"]
        for item in subobjects
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(components) != 1:
        raise RuntimeError("reloaded fast-lane Blueprint is invalid")
    manifest["glb_contract"] = glb_contract
    manifest["runtime_contract"] = _validate_assets(assets, components[0])
    manifest["reload_verification"] = {
        "status": "passed",
        "verified_at": _utc_now(),
        "process": "second_ue_commandlet",
    }
    _write_json_atomic(MANIFEST_PATH, manifest)
    spear.log(f"VERIFY_ROUTE2_TOKENRIG_FASTLANE_OK manifest={MANIFEST_PATH}")


def main():
    _validate_inputs()
    glb_contract = _read_glb_contract(SOURCE_GLB)
    if VERIFY_ONLY:
        _verify_existing(glb_contract)
        return
    if MANIFEST_PATH.exists():
        raise RuntimeError("fast-lane UE manifest already exists")
    for path in (MESH_DIR, BP_DIR):
        if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
            raise RuntimeError(f"refusing to replace existing UE directory: {path}")
    try:
        unreal.EditorAssetLibrary.make_directory(directory_path=MESH_DIR)
        unreal.EditorAssetLibrary.make_directory(directory_path=BP_DIR)
        # Interchange-backed AssetTools can return before its task graph has
        # completed.  UAssetImportTask.get_objects() is the engine-supported
        # blocking join: it calls AsyncResults->WaitUntilDone() before exposing
        # the imported objects.
        task = unreal.AssetImportTask()
        task.set_editor_property(name="async_", value=True)
        task.set_editor_property(name="automated", value=True)
        task.set_editor_property(name="destination_path", value=MESH_DIR)
        task.set_editor_property(name="filename", value=str(SOURCE_GLB))
        task.set_editor_property(name="replace_existing", value=False)
        task.set_editor_property(name="replace_existing_settings", value=False)
        task.set_editor_property(name="save", value=False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(import_tasks=[task])
        imported_objects = task.get_objects()
        if not imported_objects:
            raise RuntimeError("Interchange task completed without imported objects")
        unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
        unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
            directory_path=MESH_DIR,
            only_if_is_dirty=False,
            recursive=True,
        )
        assets = _collect_assets()
        bp_path, component = _create_blueprint(assets)
        runtime = _validate_assets(assets, component)
        manifest = {
            "schema": SCHEMA,
            "generated_at": _utc_now(),
            "tag": TAG,
            "asset_id": ASSET_ID,
            "usage_scope": USAGE_SCOPE,
            "formal_registration_authorized": False,
            "source_glb": str(SOURCE_GLB),
            "source_glb_sha256": _sha256(SOURCE_GLB),
            "source_glb_size_bytes": SOURCE_GLB.stat().st_size,
            "texture_transcode_manifest": {
                "path": str(SOURCE_GLB.parent / "ue_texture_transcode_manifest.json"),
                "sha256": _sha256(SOURCE_GLB.parent / "ue_texture_transcode_manifest.json"),
            },
            "glb_contract": glb_contract,
            "content": _content(assets, bp_path),
            "runtime_contract": runtime,
            "reload_verification": {"status": "pending"},
        }
        _write_json_atomic(MANIFEST_PATH, manifest)
    except BaseException:
        _cleanup()
        raise
    spear.log(
        f"IMPORT_ROUTE2_TOKENRIG_FASTLANE_OK tag={TAG} "
        f"bp={bp_path} manifest={MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
