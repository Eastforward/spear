"""Import one hash-locked normalized GLB into an isolated UE scale probe."""

import hashlib
import json
import os
from pathlib import Path

import unreal


SOURCE = Path(
    "/data/jzy/code/AVEngine/external/SPEAR/tmp/"
    "rocketbox_native_runtime_ue_scale_probe_v1/"
    "rocketbox_male_adult_01_original/grounded_metric_normalized.glb"
)
SOURCE_SHA256 = "0ce17511306d6ba23d3aae4d17fe7b9f775267cda7dcf28d1a35cfe54a43641a"
DESTINATION = "/Game/MyAssets/Audioset/Meshes/gate_rocketbox_native_grounded_scale_probe_v1"
EVIDENCE = SOURCE.parent / "ue_grounded_scale_probe.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


if sha256(SOURCE) != SOURCE_SHA256:
    raise RuntimeError("normalized scale-probe GLB hash changed")
if EVIDENCE.exists():
    raise RuntimeError(f"refusing to replace scale evidence: {EVIDENCE}")
if unreal.EditorAssetLibrary.does_directory_exist(DESTINATION):
    raise RuntimeError(f"refusing to replace scale probe directory: {DESTINATION}")
if not unreal.EditorAssetLibrary.make_directory(DESTINATION):
    raise RuntimeError("could not create isolated scale probe directory")
try:
    task = unreal.AssetImportTask()
    task.set_editor_property("async_", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("destination_path", DESTINATION)
    task.set_editor_property("filename", str(SOURCE))
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("replace_existing_settings", False)
    task.set_editor_property("save", False)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    if not task.get_objects():
        raise RuntimeError("scale probe import returned no objects")
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
        DESTINATION, only_if_is_dirty=False, recursive=True
    )
    meshes = [
        unreal.load_asset(path)
        for path in unreal.EditorAssetLibrary.list_assets(
            DESTINATION, recursive=True, include_folder=False
        )
        if str(
            unreal.EditorAssetLibrary.find_asset_data(path)
            .get_editor_property("asset_class_path")
            .get_editor_property("asset_name")
        )
        == "SkeletalMesh"
    ]
    if len(meshes) != 1 or meshes[0] is None:
        raise RuntimeError(f"expected one imported SkeletalMesh, got {len(meshes)}")
    bounds = meshes[0].get_imported_bounds()
    height_cm = 2.0 * float(bounds.box_extent.z)
    bottom_cm = float(bounds.origin.z - bounds.box_extent.z)
    top_cm = float(bounds.origin.z + bounds.box_extent.z)
    ground_passed = -5.0 <= bottom_cm <= 5.0
    height_passed = 165.0 <= height_cm <= 200.0
    payload = {
        "schema": "rocketbox_native_ue_scale_probe_v1",
        "status": "passed" if height_passed and ground_passed else "failed",
        "source_glb": str(SOURCE),
        "source_glb_sha256": SOURCE_SHA256,
        "destination": DESTINATION,
        "skeletal_mesh": meshes[0].get_path_name(),
        "bounds": {
            "origin_cm": xyz(bounds.origin),
            "box_extent_cm": xyz(bounds.box_extent),
            "height_cm": height_cm,
            "bottom_cm": bottom_cm,
            "top_cm": top_cm,
            "sphere_radius_cm": float(bounds.sphere_radius),
        },
        "adult_height_range_cm": [165.0, 200.0],
        "adult_height_passed": height_passed,
        "ground_bottom_range_cm": [-5.0, 5.0],
        "ground_passed": ground_passed,
    }
    temporary = EVIDENCE.with_name(f".{EVIDENCE.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.link(temporary, EVIDENCE)
    temporary.unlink()
    if payload["status"] != "passed":
        raise RuntimeError(
            f"normalized UE scale remains invalid: height={height_cm} cm, "
            f"bottom={bottom_cm} cm"
        )
    print("ROCKETBOX_NATIVE_SCALE_PROBE=" + json.dumps(payload, sort_keys=True))
except BaseException:
    if EVIDENCE.exists():
        EVIDENCE.unlink()
    unreal.EditorAssetLibrary.delete_directory(DESTINATION)
    raise
