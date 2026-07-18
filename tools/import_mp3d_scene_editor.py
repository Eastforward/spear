"""Import the fair MP3D scene and exact Habitat-M2 Beagle into Unreal.

This script runs inside the UE 5.5 editor through ``tools/run_editor_script.py``.
It owns exactly three isolated content directories and never deletes assets
outside them:

* ``/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy``
* ``/Game/MyAssets/Audioset/Meshes/gate_m2_beagle_v7_world_contact_r5``
* ``/Game/MyAssets/Audioset/Blueprints/gate_m2_beagle_v7_world_contact_r5``

Required environment variables:

``MP3D_UE_PREPARED_MANIFEST``
    Manifest written by :mod:`tools.prepare_mp3d_ue_scene`.
``MP3D_UE_M2_BEAGLE_MANIFEST``
    Habitat M2 ``asset_manifest.json`` for the exact Beagle runtime.
``MP3D_UE_IMPORT_RESULT``
    External JSON evidence path.

Set ``MP3D_UE_REPLACE_EXISTING=1`` only when intentionally replacing these
three exact directories.  Set ``MP3D_UE_VERIFY_ONLY=1`` for a second-process
reload/readback that updates ``reload_verification`` to ``passed``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spear
import unreal


SCHEMA = "avengine_mp3d_ue_import_result_v1"
PREPARED_SCHEMA = "avengine_mp3d_ue_prepared_scene_v1"
M2_SCHEMA = "avengine_animal_asset_package_v1"
SCENE_ID = "17DRP5sb8fy"
EXPECTED_RAW_MP3D_SHA256 = (
    "334456925e056c83a9a7a5c768b3d37cdd23425d8ca20743bfce015be3f56b04"
)
SCENE_CONTENT_DIR = "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy"
BEAGLE_TAG = "m2_beagle_v7_world_contact_r5"
BEAGLE_MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{BEAGLE_TAG}"
BEAGLE_BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{BEAGLE_TAG}"
BEAGLE_BP_NAME = f"BP_gate_{BEAGLE_TAG}"
BEAGLE_BP_PACKAGE_PATH = f"{BEAGLE_BP_DIR}/{BEAGLE_BP_NAME}"
# Freeze the formal comparison binding as an auditable literal.  A package
# rename must update this independently checked contract instead of silently
# following the nearby derived names.
BEAGLE_BP_CLASS_PATH = (
    "/Game/MyAssets/Audioset/Blueprints/gate_m2_beagle_v7_world_contact_r5/"
    "BP_gate_m2_beagle_v7_world_contact_r5."
    "BP_gate_m2_beagle_v7_world_contact_r5_C"
)
EXPECTED_M2_BEAGLE_SHA256 = (
    "788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c"
)
EXPECTED_M2_BEAGLE_SIZE_BYTES = 16_219_232
EXPECTED_M2_BEAGLE_ASSET_ID = (
    "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
)
EXPECTED_SCENE_STATIC_MESH_COUNT = 71
EXPECTED_SCENE_MATERIAL_COUNT = 23
EXPECTED_SCENE_TEXTURE_COUNT = 23
EXPECTED_UE_SCENE_BOUNDS_CM = {
    "minimum": [-1159.3440055847168, -539.2021179199219, -12.755300104618073],
    "maximum": [475.7026195526123, 288.6620044708252, 267.87829399108887],
}
UE_SCENE_BOUNDS_TOLERANCE_CM = 0.5
EXPECTED_BEAGLE_ANIMATIONS = {"Idle", "Walking"}
MANAGED_CONTENT_DIRS = (
    SCENE_CONTENT_DIR,
    BEAGLE_MESH_DIR,
    BEAGLE_BP_DIR,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} is missing or not a direct file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} root must be an object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_managed_directory(path: str) -> None:
    if path not in MANAGED_CONTENT_DIRS:
        raise RuntimeError(f"refusing to manage non-owned UE directory: {path}")
    if ".." in path or not path.startswith("/Game/MyAssets/Audioset/"):
        raise RuntimeError(f"unsafe UE content directory: {path}")


def _remove_managed_directory(path: str) -> None:
    _validate_managed_directory(path)
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        if not unreal.EditorAssetLibrary.delete_directory(directory_path=path):
            raise RuntimeError(f"failed to delete owned UE directory: {path}")


def _make_managed_directory(path: str) -> None:
    _validate_managed_directory(path)
    if not unreal.EditorAssetLibrary.make_directory(directory_path=path):
        raise RuntimeError(f"failed to create owned UE directory: {path}")


def _validate_inputs() -> dict[str, Any]:
    prepared_manifest_path = Path(
        os.environ["MP3D_UE_PREPARED_MANIFEST"]
    ).resolve()
    m2_manifest_path = Path(os.environ["MP3D_UE_M2_BEAGLE_MANIFEST"]).resolve()
    result_path = Path(os.environ["MP3D_UE_IMPORT_RESULT"]).resolve()
    prepared_manifest = _load_json(prepared_manifest_path, "prepared MP3D manifest")
    if (
        prepared_manifest.get("schema") != PREPARED_SCHEMA
        or prepared_manifest.get("status") != "passed"
        or prepared_manifest.get("scene_id") != SCENE_ID
        or prepared_manifest.get("source", {}).get("sha256")
        != EXPECTED_RAW_MP3D_SHA256
        or prepared_manifest.get("root_mesh_identity", {}).get("status") != "passed"
        or prepared_manifest.get("root_mesh_identity", {}).get("mesh_count")
        != EXPECTED_SCENE_STATIC_MESH_COUNT
        or prepared_manifest.get("reference_bounds_validation", {}).get("status")
        != "passed"
    ):
        raise RuntimeError("prepared MP3D manifest contract changed")
    prepared_glb = Path(prepared_manifest["prepared"]["path"]).resolve()
    source_glb = Path(prepared_manifest["source"]["path"]).resolve()
    for path, record, label in (
        (prepared_glb, prepared_manifest["prepared"], "prepared MP3D GLB"),
        (source_glb, prepared_manifest["source"], "raw MP3D GLB"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise RuntimeError(f"{label} authentication failed: {path}")

    m2_manifest = _load_json(m2_manifest_path, "Habitat M2 Beagle manifest")
    visual_records = [
        item
        for item in m2_manifest.get("files", [])
        if item.get("role") == "visual"
    ]
    if (
        m2_manifest.get("schema") != M2_SCHEMA
        or m2_manifest.get("asset_id") != EXPECTED_M2_BEAGLE_ASSET_ID
        or m2_manifest.get("admission_state") != "canary_qualified"
        or m2_manifest.get("qualification", {}).get("automatic_qa_status")
        != "pass"
        or m2_manifest.get("qualification", {}).get("human_visual_review_status")
        != "pass"
        or len(visual_records) != 1
        or visual_records[0].get("path") != "visual.glb"
        or visual_records[0].get("sha256") != EXPECTED_M2_BEAGLE_SHA256
        or visual_records[0].get("byte_size") != EXPECTED_M2_BEAGLE_SIZE_BYTES
    ):
        raise RuntimeError("Habitat M2 Beagle manifest contract changed")
    m2_glb = (m2_manifest_path.parent / visual_records[0]["path"]).resolve()
    if (
        not m2_glb.is_file()
        or m2_glb.is_symlink()
        or _sha256(m2_glb) != EXPECTED_M2_BEAGLE_SHA256
        or m2_glb.stat().st_size != EXPECTED_M2_BEAGLE_SIZE_BYTES
    ):
        raise RuntimeError("exact Habitat M2 Beagle visual.glb authentication failed")

    return {
        "prepared_manifest_path": prepared_manifest_path,
        "prepared_manifest": prepared_manifest,
        "prepared_glb": prepared_glb,
        "raw_glb": source_glb,
        "m2_manifest_path": m2_manifest_path,
        "m2_manifest": m2_manifest,
        "m2_glb": m2_glb,
        "result_path": result_path,
    }


def _asset_class_name(asset_path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_path = data.get_editor_property(name="asset_class_path")
    return str(class_path.get_editor_property(name="asset_name"))


def _asset_name(asset_path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    return str(data.get_editor_property(name="asset_name"))


def _collect_assets(directory: str) -> dict[str, list[str]]:
    _validate_managed_directory(directory)
    paths = sorted(
        str(path)
        for path in unreal.EditorAssetLibrary.list_assets(
            directory_path=directory,
            recursive=True,
            include_folder=False,
        )
    )
    if not paths or any(not path.startswith(f"{directory}/") for path in paths):
        raise RuntimeError(f"UE assets escaped or are missing from {directory}")
    result: dict[str, list[str]] = {}
    for path in paths:
        result.setdefault(_asset_class_name(path), []).append(path)
    return {key: sorted(value) for key, value in sorted(result.items())}


def _class_paths(
    assets: dict[str, list[str]], class_names: set[str]
) -> list[str]:
    return sorted(
        path
        for class_name in class_names
        for path in assets.get(class_name, [])
    )


def _vector_xyz(value: Any) -> list[float]:
    result = []
    for name in ("x", "y", "z"):
        try:
            component = value.get_editor_property(name=name)
        except Exception:
            component = getattr(value, name)
        result.append(float(component))
    return result


def _validate_scene_bounds(static_mesh_paths: list[str]) -> dict[str, Any]:
    records = []
    aggregate_minimum = [float("inf")] * 3
    aggregate_maximum = [float("-inf")] * 3
    for path in static_mesh_paths:
        mesh = unreal.load_asset(name=path)
        if mesh is None or not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"failed to load imported MP3D StaticMesh: {path}")
        bounds = mesh.get_bounding_box()
        minimum = _vector_xyz(bounds.get_editor_property(name="min"))
        maximum = _vector_xyz(bounds.get_editor_property(name="max"))
        if any(not math.isfinite(value) for value in (*minimum, *maximum)) or any(
            minimum[axis] > maximum[axis] for axis in range(3)
        ):
            raise RuntimeError(f"invalid imported StaticMesh bounds: {path}")
        for axis in range(3):
            aggregate_minimum[axis] = min(aggregate_minimum[axis], minimum[axis])
            aggregate_maximum[axis] = max(aggregate_maximum[axis], maximum[axis])
        records.append({"object_path": path, "minimum_cm": minimum, "maximum_cm": maximum})
    maximum_error = max(
        abs(observed - EXPECTED_UE_SCENE_BOUNDS_CM[key][axis])
        for key, observed_values in (
            ("minimum", aggregate_minimum),
            ("maximum", aggregate_maximum),
        )
        for axis, observed in enumerate(observed_values)
    )
    if maximum_error > UE_SCENE_BOUNDS_TOLERANCE_CM:
        raise RuntimeError(
            "imported MP3D UE bounds do not match the pinned axis/scale contract: "
            f"min={aggregate_minimum} max={aggregate_maximum} error={maximum_error}"
        )
    return {
        "status": "passed",
        "semantics": "aggregate StaticMesh local-space bounds after UE Interchange",
        "minimum_cm": aggregate_minimum,
        "maximum_cm": aggregate_maximum,
        "expected": EXPECTED_UE_SCENE_BOUNDS_CM,
        "maximum_absolute_error_cm": maximum_error,
        "tolerance_cm": UE_SCENE_BOUNDS_TOLERANCE_CM,
        "per_mesh": records,
    }


def _import_glb(source_glb: Path, destination: str) -> None:
    task = unreal.AssetImportTask()
    task.set_editor_property(name="async_", value=True)
    task.set_editor_property(name="automated", value=True)
    task.set_editor_property(name="destination_path", value=destination)
    task.set_editor_property(name="filename", value=str(source_glb))
    task.set_editor_property(name="replace_existing", value=False)
    task.set_editor_property(name="replace_existing_settings", value=False)
    task.set_editor_property(name="save", value=False)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(
        import_tasks=[task]
    )
    # Interchange can return while its task graph is active. get_objects() is
    # the engine-backed join used by the existing verified importers.
    if not task.get_objects():
        raise RuntimeError(f"Interchange imported no objects from {source_glb}")
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
        directory_path=destination,
        only_if_is_dirty=False,
        recursive=True,
    )


def _validate_scene_assets(assets: dict[str, list[str]]) -> dict[str, Any]:
    static_meshes = assets.get("StaticMesh", [])
    materials = _class_paths(assets, {"Material", "MaterialInstanceConstant"})
    textures = _class_paths(
        assets, {"Texture2D", "TextureCube", "VirtualTexture2D"}
    )
    if len(static_meshes) != EXPECTED_SCENE_STATIC_MESH_COUNT:
        raise RuntimeError(
            f"expected 71 MP3D StaticMesh assets, observed {len(static_meshes)}"
        )
    if len(materials) < EXPECTED_SCENE_MATERIAL_COUNT:
        raise RuntimeError(
            f"expected at least 23 MP3D materials, observed {len(materials)}"
        )
    if len(textures) < EXPECTED_SCENE_TEXTURE_COUNT:
        raise RuntimeError(
            f"expected at least 23 MP3D textures, observed {len(textures)}"
        )
    bounds = _validate_scene_bounds(static_meshes)
    return {
        "status": "passed",
        "content_directory": SCENE_CONTENT_DIR,
        "object_count": sum(len(value) for value in assets.values()),
        "class_counts": {key: len(value) for key, value in assets.items()},
        "all_object_paths": sorted(
            path for value in assets.values() for path in value
        ),
        "static_meshes": static_meshes,
        "materials": materials,
        "textures": textures,
        "static_mesh_count": len(static_meshes),
        "material_count": len(materials),
        "texture_count": len(textures),
        "ue_bounds": bounds,
    }


def _create_beagle_blueprint(
    skeletal_mesh_path: str,
    animation_paths: dict[str, str],
) -> None:
    blueprint = spear.editor.create_blueprint_asset(
        asset_name=BEAGLE_BP_NAME,
        package_dir=BEAGLE_BP_DIR,
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
        raise RuntimeError("exact M2 Beagle Blueprint has no unique mesh component")
    component = components[0]
    component.set_animation_mode(
        animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE
    )
    component.set_skeletal_mesh_asset(new_mesh=unreal.load_asset(skeletal_mesh_path))
    component.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=unreal.load_asset(animation_paths["Walking"]),
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


def _loaded_beagle_component() -> Any:
    blueprint = unreal.load_asset(name=BEAGLE_BP_PACKAGE_PATH)
    if blueprint is None:
        raise RuntimeError(f"failed to load exact M2 Beagle BP: {BEAGLE_BP_PACKAGE_PATH}")
    components = [
        item["object"]
        for item in spear.editor.get_subobject_descs_for_blueprint_asset(
            blueprint_asset=blueprint
        )
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(components) != 1:
        raise RuntimeError("reloaded exact M2 Beagle BP has no unique mesh component")
    return components[0]


def _validate_beagle_assets(assets: dict[str, list[str]]) -> dict[str, Any]:
    skeletal_meshes = assets.get("SkeletalMesh", [])
    skeletons = assets.get("Skeleton", [])
    animation_paths = {
        _asset_name(path): path for path in assets.get("AnimSequence", [])
    }
    materials = _class_paths(assets, {"Material", "MaterialInstanceConstant"})
    textures = _class_paths(
        assets, {"Texture2D", "TextureCube", "VirtualTexture2D"}
    )
    if len(skeletal_meshes) != 1 or len(skeletons) != 1:
        raise RuntimeError(
            "exact M2 Beagle import requires one SkeletalMesh and one Skeleton"
        )
    if set(animation_paths) != EXPECTED_BEAGLE_ANIMATIONS:
        raise RuntimeError(
            f"exact M2 Beagle animations changed: {sorted(animation_paths)}"
        )
    if not materials or not textures:
        raise RuntimeError("exact M2 Beagle import lost material or texture assets")
    if not unreal.EditorAssetLibrary.does_asset_exist(
        asset_path=BEAGLE_BP_PACKAGE_PATH
    ):
        raise RuntimeError(
            "exact M2 Beagle Blueprint is missing; validation never recreates it"
        )
    component = _loaded_beagle_component()
    bone_count = int(component.get_num_bones())
    if bone_count != 35:
        raise RuntimeError(f"exact M2 Beagle bone count {bone_count} != 35")
    return {
        "status": "passed",
        "tag": BEAGLE_TAG,
        "mesh_content_directory": BEAGLE_MESH_DIR,
        "blueprint_content_directory": BEAGLE_BP_DIR,
        "blueprint_package_path": BEAGLE_BP_PACKAGE_PATH,
        "blueprint_class_path": BEAGLE_BP_CLASS_PATH,
        "skeletal_mesh": skeletal_meshes[0],
        "skeleton": skeletons[0],
        "bone_count": bone_count,
        "animations": animation_paths,
        "materials": materials,
        "textures": textures,
        "all_object_paths": sorted(
            path for value in assets.values() for path in value
        )
        + [BEAGLE_BP_PACKAGE_PATH],
    }


def _build_result(
    inputs: dict[str, Any],
    scene: dict[str, Any],
    beagle: dict[str, Any],
) -> dict[str, Any]:
    prepared = inputs["prepared_manifest"]
    return {
        "schema": SCHEMA,
        "generated_at": _utc_now(),
        "status": "passed",
        "scene_id": SCENE_ID,
        "owned_content_directories": list(MANAGED_CONTENT_DIRS),
        "replacement_policy": {
            "only_exact_owned_directories_may_be_replaced": True,
            "replace_existing_requested": os.environ.get(
                "MP3D_UE_REPLACE_EXISTING", "0"
            )
            == "1",
        },
        "source": prepared["source"],
        "prepared": {
            **prepared["prepared"],
            "manifest_path": str(inputs["prepared_manifest_path"]),
            "manifest_sha256": _sha256(inputs["prepared_manifest_path"]),
        },
        "coordinate_contract": prepared["coordinate_contract"],
        "scene_content": scene,
        "m2_beagle": {
            "source": {
                "path": str(inputs["m2_glb"]),
                "sha256": EXPECTED_M2_BEAGLE_SHA256,
                "size_bytes": EXPECTED_M2_BEAGLE_SIZE_BYTES,
            },
            "source_manifest": {
                "path": str(inputs["m2_manifest_path"]),
                "sha256": _sha256(inputs["m2_manifest_path"]),
                "size_bytes": inputs["m2_manifest_path"].stat().st_size,
            },
            "source_asset_id": EXPECTED_M2_BEAGLE_ASSET_ID,
            "content": beagle,
            "exact_habitat_m2_runtime": True,
        },
        "reload_verification": {
            "status": "pending",
            "required_before_formal_comparison_render": True,
        },
        "claim_boundary": (
            "This proves UE editor import/readback for the canonical MP3D "
            "scene and exact Habitat M2 Beagle source. Cooked PAK presence and "
            "packaged runtime rendering require separate evidence."
        ),
    }


def _verify_only(inputs: dict[str, Any]) -> None:
    result_path = inputs["result_path"]
    result = _load_json(result_path, "MP3D UE import result")
    if (
        result.get("schema") != SCHEMA
        or result.get("status") != "passed"
        or result.get("scene_id") != SCENE_ID
        or result.get("prepared", {}).get("sha256")
        != inputs["prepared_manifest"]["prepared"]["sha256"]
        or result.get("source", {}).get("sha256")
        != inputs["prepared_manifest"]["source"]["sha256"]
        or result.get("m2_beagle", {}).get("source", {}).get("sha256")
        != EXPECTED_M2_BEAGLE_SHA256
    ):
        raise RuntimeError("MP3D UE import result changed before reload verification")
    scene = _validate_scene_assets(_collect_assets(SCENE_CONTENT_DIR))
    beagle = _validate_beagle_assets(_collect_assets(BEAGLE_MESH_DIR))
    if scene["static_meshes"] != result["scene_content"]["static_meshes"]:
        raise RuntimeError("MP3D StaticMesh object paths changed after reload")
    for key in ("minimum_cm", "maximum_cm"):
        before = result["scene_content"].get("ue_bounds", {}).get(key)
        after = scene["ue_bounds"][key]
        if not isinstance(before, list) or len(before) != 3 or any(
            abs(float(left) - float(right)) > UE_SCENE_BOUNDS_TOLERANCE_CM
            for left, right in zip(before, after)
        ):
            raise RuntimeError(f"MP3D StaticMesh aggregate {key} changed after reload")
    if beagle["skeletal_mesh"] != result["m2_beagle"]["content"]["skeletal_mesh"]:
        raise RuntimeError("exact M2 Beagle SkeletalMesh changed after reload")
    if beagle["animations"] != result["m2_beagle"]["content"]["animations"]:
        raise RuntimeError("exact M2 Beagle animations changed after reload")
    result["scene_content"] = scene
    result["m2_beagle"]["content"] = beagle
    result["reload_verification"] = {
        "status": "passed",
        "verified_at": _utc_now(),
        "process": "second_ue_editor_process",
        "required_before_formal_comparison_render": True,
    }
    _atomic_write_json(result_path, result)
    spear.log(f"VERIFY_MP3D_UE_IMPORT_OK result={result_path}")


def main() -> None:
    inputs = _validate_inputs()
    if os.environ.get("MP3D_UE_VERIFY_ONLY", "0") == "1":
        _verify_only(inputs)
        return

    replace_existing = os.environ.get("MP3D_UE_REPLACE_EXISTING", "0") == "1"
    existing_directories = [
        path
        for path in MANAGED_CONTENT_DIRS
        if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path)
    ]
    if existing_directories and not replace_existing:
        raise RuntimeError(
            "owned UE content already exists; set MP3D_UE_REPLACE_EXISTING=1 "
            f"to replace only these directories: {existing_directories}"
        )
    if inputs["result_path"].exists() and not replace_existing:
        raise RuntimeError(
            f"UE import result already exists: {inputs['result_path']}"
        )

    created: list[str] = []
    try:
        if replace_existing:
            for directory in MANAGED_CONTENT_DIRS:
                _remove_managed_directory(directory)
        for directory in MANAGED_CONTENT_DIRS:
            _make_managed_directory(directory)
            created.append(directory)

        _import_glb(inputs["prepared_glb"], SCENE_CONTENT_DIR)
        scene = _validate_scene_assets(_collect_assets(SCENE_CONTENT_DIR))

        _import_glb(inputs["m2_glb"], BEAGLE_MESH_DIR)
        beagle_assets = _collect_assets(BEAGLE_MESH_DIR)
        # Creation happens only after all imported skeletal/action assets have
        # passed their exact source and class readback.
        skeletal_meshes = beagle_assets.get("SkeletalMesh", [])
        animation_paths = {
            _asset_name(path): path
            for path in beagle_assets.get("AnimSequence", [])
        }
        if len(skeletal_meshes) != 1 or set(animation_paths) != EXPECTED_BEAGLE_ANIMATIONS:
            raise RuntimeError("exact M2 Beagle imported classes/actions are invalid")
        _create_beagle_blueprint(skeletal_meshes[0], animation_paths)
        beagle = _validate_beagle_assets(beagle_assets)

        result = _build_result(inputs, scene, beagle)
        _atomic_write_json(inputs["result_path"], result)
    except BaseException:
        for directory in reversed(created):
            _remove_managed_directory(directory)
        raise

    spear.log(
        "IMPORT_MP3D_UE_SCENE_OK "
        f"static_meshes={scene['static_mesh_count']} "
        f"beagle_sha256={EXPECTED_M2_BEAGLE_SHA256} "
        f"result={inputs['result_path']}"
    )


if __name__ == "__main__":
    main()
    try:
        unreal.SystemLibrary.quit_editor()
    except Exception as error:
        spear.log(f"WARN explicit UE editor exit request failed: {error}")
