"""Headless UE Editor Python script: import a per-tag rigged GLB (from
robust_skin_transfer) and wrap it in a Blueprint that plays the Walking
anim on loop.

Structurally a tag-parameterized clone of import_animated_dog_editor.py.
Reads env vars:
  GATE_TAG          - species tag, e.g. "cat_persian"
  GATE_RIGGED_GLB   - absolute path to the rigged skeletal-mesh GLB
  GATE_IMPORT_JOB_JSON
      Required for ``pixal_*`` tags.  Authenticated per-job identity emitted
      by ``import_pixal_animal_batch_editor.py``.

Emits Blueprint at /Game/MyAssets/Audioset/Blueprints/gate_{TAG}/BP_gate_{TAG}.
"""

import hashlib
import json
import os
import posixpath
import re
import struct
import sys
from pathlib import Path

import spear
import unreal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts  # noqa: E402


EXPECTED_ACTIONS = ["Idle", "Walking"]
PIXAL_JOB_TYPE = "user_approved_generated_animal"
IMPORT_JOB_IDENTITY_SCHEMA = "pixal_animal_ue_import_job_identity_v1"
IMPORT_JOB_RECEIPT_SCHEMA = "pixal_animal_ue_import_job_receipt_v1"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
CANONICAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
JOB_IDENTITY_FIELDS = {
    "schema",
    "job_type",
    "asset_id",
    "legacy_tag",
    "tag",
    "expected_actions",
    "rigged_glb",
    "rigged_glb_sha256",
    "input_manifest_sha256",
    "batch_sha256",
    "job_identity_sha256",
}


def _required_environment(name):
    value = os.environ.get(name)
    if value is None or not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


TAG = _required_environment("GATE_TAG")
RIGGED_GLB = _required_environment("GATE_RIGGED_GLB")
MESH_CONTENT_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_CONTENT_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"
FUR_MAT_HINT = (
    "fur"  # blender_robust_swap_mesh_keep_rig names its material "RobustSwap_Fur"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"value is not canonical JSON: {error}") from error


def _load_job_identity(value):
    try:
        payload = contracts.strict_json_loads(value)
    except contracts.StrictJSONError as error:
        raise RuntimeError("GATE_IMPORT_JOB_JSON is not strict JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("GATE_IMPORT_JOB_JSON must be a JSON object")
    return payload


def _read_glb_document(path, expected_sha256):
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"rigged GLB is not readable: {path}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("external rigged GLB SHA-256 pin mismatched")
    if len(raw) < 20:
        raise RuntimeError(f"rigged GLB is incomplete: {path}")
    magic, version, declared_size = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_size != len(raw):
        raise RuntimeError(f"rigged GLB header is invalid: {path}")

    offset = 12
    json_chunks = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise RuntimeError(f"rigged GLB chunk header is truncated: {path}")
        chunk_size, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_end = offset + chunk_size
        if chunk_size % 4 != 0 or chunk_end > len(raw):
            raise RuntimeError(f"rigged GLB chunk is invalid: {path}")
        if chunk_type == 0x4E4F534A:
            json_chunks.append(raw[offset:chunk_end])
        offset = chunk_end
    if offset != len(raw) or len(json_chunks) != 1:
        raise RuntimeError(f"rigged GLB must contain exactly one JSON chunk: {path}")
    try:
        document = contracts.strict_json_loads(json_chunks[0].rstrip(b" \t\r\n\x00"))
    except contracts.StrictJSONError as error:
        raise RuntimeError(f"rigged GLB JSON document is invalid: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"rigged GLB JSON document must be an object: {path}")
    return document


def _validate_prewrite_contract():
    if len(TAG) > 104 or SAFE_TAG_PATTERN.fullmatch(TAG) is None or TAG.lower() != TAG:
        raise RuntimeError(f"unsafe or non-normalized UE gate tag: {TAG!r}")
    raw_source = Path(RIGGED_GLB)
    if (
        not raw_source.is_absolute()
        or raw_source.is_symlink()
        or not raw_source.is_file()
    ):
        raise RuntimeError(f"rigged GLB must be an absolute, direct file: {raw_source}")
    source = raw_source.resolve()
    source_sha256 = _sha256(source)
    raw_identity = os.environ.get("GATE_IMPORT_JOB_JSON")
    if raw_identity is None:
        if TAG.startswith("pixal_"):
            raise RuntimeError(
                "pixal_* UE writes require authenticated GATE_IMPORT_JOB_JSON"
            )
        _read_glb_document(source, source_sha256)
        return source, hashlib.sha256(
            _canonical_json(
                {
                    "tag": TAG,
                    "rigged_glb": str(source),
                    "rigged_glb_sha256": source_sha256,
                }
            ).encode("utf-8")
        ).hexdigest()

    identity = _load_job_identity(raw_identity)
    asset_id = identity.get("asset_id")
    if (
        set(identity) != JOB_IDENTITY_FIELDS
        or identity.get("schema") != IMPORT_JOB_IDENTITY_SCHEMA
        or identity.get("job_type") != PIXAL_JOB_TYPE
        or not isinstance(asset_id, str)
        or len(asset_id) > 96
        or CANONICAL_ID_PATTERN.fullmatch(asset_id) is None
        or identity.get("legacy_tag") != asset_id
        or identity.get("tag") != f"pixal_{asset_id}"
        or identity.get("tag") != TAG
        or identity.get("expected_actions") != EXPECTED_ACTIONS
        or identity.get("rigged_glb") != str(source)
        or identity.get("rigged_glb_sha256") != source_sha256
    ):
        raise RuntimeError("authenticated per-job UE import identity mismatched")
    for field in (
        "rigged_glb_sha256",
        "input_manifest_sha256",
        "batch_sha256",
        "job_identity_sha256",
    ):
        value = identity.get(field)
        if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
            raise RuntimeError(f"per-job UE import {field} is not a SHA-256")
    _read_glb_document(source, identity["rigged_glb_sha256"])
    return source, identity["job_identity_sha256"]


def _assert_content_targets_absent():
    existing = [
        path
        for path in (MESH_CONTENT_DIR, BP_CONTENT_DIR)
        if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path)
    ]
    bp_path = posixpath.join(BP_CONTENT_DIR, BP_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        existing.append(bp_path)
    if existing:
        raise RuntimeError(
            "refusing to replace existing UE content target(s): "
            + ", ".join(sorted(set(existing)))
        )


def _make_or_clear_dir(path):
    """Create a new UE directory; retained name is compatibility-only."""
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        raise RuntimeError(f"refusing to replace existing UE directory: {path}")
    if not unreal.EditorAssetLibrary.make_directory(directory_path=path):
        raise RuntimeError(f"could not create new UE directory: {path}")


def _asset_details(asset_path):
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_path = data.get_editor_property(name="asset_class_path")
    class_name = str(class_path.get_editor_property(name="asset_name"))
    package_dir = str(data.get_editor_property(name="package_path"))
    asset_name = str(data.get_editor_property(name="asset_name"))
    return (
        class_name,
        asset_name,
        posixpath.join(package_dir, f"{asset_name}.{asset_name}"),
    )


def _blueprint_component(blueprint_asset):
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint_asset
    )
    actors = [
        item["object"]
        for item in subobjects
        if isinstance(item.get("object"), unreal.SkeletalMeshActor)
    ]
    components = [
        item["object"]
        for item in subobjects
        if isinstance(item.get("object"), unreal.SkeletalMeshComponent)
    ]
    if len(subobjects) != 2 or len(actors) != 1 or len(components) != 1:
        raise RuntimeError(
            "Blueprint must contain exactly one SkeletalMeshActor and "
            "one SkeletalMeshComponent"
        )
    return components[0]


def _object_path(value):
    if value is None:
        return None
    return str(value.get_path_name())


def main():
    source, job_identity_sha256 = _validate_prewrite_contract()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    editor_asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)

    spear.log(f"Importing glb: {source}")
    spear.log(f"  -> mesh content dir: {MESH_CONTENT_DIR}")
    spear.log(f"  -> BP content dir:   {BP_CONTENT_DIR}")

    # Preflight both destinations before creating either one.  The helper
    # repeats the check at creation time so a pre-existing target can never be
    # cleared, even if this script is called without the batch wrapper.
    _assert_content_targets_absent()
    _make_or_clear_dir(MESH_CONTENT_DIR)
    _make_or_clear_dir(BP_CONTENT_DIR)

    task = unreal.AssetImportTask()
    # UE 5.5's Interchange-backed GLB import may return from
    # import_asset_tasks() before the import graph has published its assets,
    # even when ``async_`` is false.  AssetImportTask.get_objects() is the
    # engine-supported blocking join: it waits for AsyncResults to finish
    # before exposing the imported objects.  Without this join, a cold editor
    # can observe an empty destination and falsely report that a valid GLB has
    # no SkeletalMesh.
    task.set_editor_property(name="async_", value=True)
    task.set_editor_property(name="automated", value=True)
    task.set_editor_property(name="destination_path", value=MESH_CONTENT_DIR)
    task.set_editor_property(name="filename", value=str(source))
    task.set_editor_property(name="replace_existing", value=False)
    task.set_editor_property(name="replace_existing_settings", value=False)
    task.set_editor_property(name="save", value=False)
    asset_tools.import_asset_tasks(import_tasks=[task])
    imported_objects = task.get_objects()
    if not imported_objects:
        raise RuntimeError(
            f"Interchange completed without imported objects for {source}"
        )
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    if not editor_asset_subsystem.save_directory(
        directory_path=MESH_CONTENT_DIR,
        only_if_is_dirty=False,
        recursive=True,
    ):
        raise RuntimeError(
            f"failed to persist imported mesh directory: {MESH_CONTENT_DIR}"
        )
    if not unreal.EditorAssetLibrary.does_directory_exist(
        directory_path=MESH_CONTENT_DIR
    ):
        raise RuntimeError(
            f"imported mesh directory disappeared before readback: {MESH_CONTENT_DIR}"
        )

    imported_paths = unreal.EditorAssetLibrary.list_assets(
        directory_path=MESH_CONTENT_DIR
    )
    spear.log("Imported assets:")
    for ap in imported_paths:
        spear.log(f"  {ap}")

    skeletal_mesh_paths = []
    animation_paths = {}
    fur_material_path = None
    for ap in imported_paths:
        cls, name, obj_path = _asset_details(ap)
        if cls == "SkeletalMesh":
            skeletal_mesh_paths.append(obj_path)
            spear.log(f"  found SkeletalMesh: {obj_path}")
        elif cls == "AnimSequence":
            if name in animation_paths:
                raise RuntimeError(f"duplicate AnimSequence action: {name}")
            animation_paths[name] = obj_path
            spear.log(f"  found AnimSequence {name}: {obj_path}")
        elif (
            cls in ("Material", "MaterialInstanceConstant")
            and FUR_MAT_HINT in name.lower()
        ):
            fur_material_path = obj_path
            spear.log(f"  found fur material: {obj_path} (cls={cls})")

    if len(skeletal_mesh_paths) != 1:
        raise RuntimeError(
            f"import must produce exactly one SkeletalMesh; found={skeletal_mesh_paths}"
        )
    if set(animation_paths) != set(EXPECTED_ACTIONS) or len(animation_paths) != len(
        EXPECTED_ACTIONS
    ):
        raise RuntimeError(
            "import must produce exactly the unique Idle and Walking "
            f"AnimSequences; found={sorted(animation_paths)}"
        )
    skeletal_mesh_path = skeletal_mesh_paths[0]
    walking_anim_path = animation_paths["Walking"]
    skeletal_mesh = unreal.load_asset(name=skeletal_mesh_path)
    walking_anim = unreal.load_asset(name=walking_anim_path)
    if skeletal_mesh is None or walking_anim is None:
        raise RuntimeError(
            "exact SkeletalMesh/Walking assets could not be loaded for Blueprint"
        )

    # Bind the fur material into slot 0 so the textured mesh survives cook.
    if fur_material_path is not None:
        skm_obj = skeletal_mesh
        fur_mat = unreal.load_asset(name=fur_material_path)
        if fur_mat is None:
            raise RuntimeError(f"fur material could not be loaded: {fur_material_path}")
        materials = skm_obj.get_editor_property("materials")
        if len(materials) > 0:
            materials[0].material_interface = fur_mat
            skm_obj.set_editor_property("materials", materials)
            if not editor_asset_subsystem.save_loaded_asset(asset_to_save=skm_obj):
                raise RuntimeError(
                    f"failed to persist material binding on {skeletal_mesh_path}"
                )
            spear.log(f"  bound {fur_material_path} to SKM slot 0 and saved")

        # Ensure baseColorTexture points at the correct Texture2D and clear
        # the baseColorFactor to white so the raw Hunyuan diffuse survives.
        diffuse_path = None
        for ap in imported_paths:
            cls, name, obj_path = _asset_details(ap)
            if cls == "Texture2D":
                diffuse_path = obj_path
                spear.log(f"  candidate Texture2D: {diffuse_path}")
                break
        if diffuse_path is not None:
            tex = unreal.load_asset(name=diffuse_path)
            for pname in ("baseColorTexture", "BaseColor", "DiffuseColor"):
                try:
                    ok = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                        instance=fur_mat,
                        parameter_name=pname,
                        value=tex,
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
                        instance=fur_mat,
                        parameter_name=pname,
                        value=tint,
                    )
                    spear.log(f"  set factor {pname!r} -> white passthrough")
                except Exception:
                    pass
            for pname, val in (
                ("metallicFactor", 0.0),
                ("MetallicFactor", 0.0),
                ("roughnessFactor", 0.95),
                ("RoughnessFactor", 0.95),
            ):
                try:
                    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
                        instance=fur_mat,
                        parameter_name=pname,
                        value=val,
                    )
                except Exception:
                    pass
            if not editor_asset_subsystem.save_loaded_asset(asset_to_save=fur_mat):
                raise RuntimeError(
                    f"failed to persist fur material: {fur_material_path}"
                )

    bp_path = posixpath.join(BP_CONTENT_DIR, BP_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        raise RuntimeError(f"refusing to replace existing UE asset: {bp_path}")

    spear.log(f"Creating BP: {bp_path}")
    blueprint_asset = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME,
        package_dir=BP_CONTENT_DIR,
        parent_class=unreal.SkeletalMeshActor,
    )
    if blueprint_asset is None:
        raise RuntimeError(f"Blueprint creation returned no asset: {bp_path}")
    smc = _blueprint_component(blueprint_asset)
    smc.set_animation_mode(animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    smc.set_skeletal_mesh_asset(new_mesh=skeletal_mesh)
    smc.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=walking_anim,
            saved_position=0.0,
            saved_play_rate=1.0,
        ),
    )
    smc.set_editor_property(
        name="visibility_based_anim_tick_option",
        value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )

    spear.log(f"Saving BP: {bp_path}")
    if not editor_asset_subsystem.save_loaded_asset(asset_to_save=blueprint_asset):
        raise RuntimeError(f"failed to persist Blueprint: {bp_path}")
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        raise RuntimeError(f"saved Blueprint is absent at readback: {bp_path}")
    saved_blueprint = unreal.load_asset(name=bp_path)
    if saved_blueprint is None:
        raise RuntimeError(f"saved Blueprint could not be loaded: {bp_path}")
    readback_component = _blueprint_component(saved_blueprint)
    readback_mesh = readback_component.get_skeletal_mesh_asset()
    readback_animation_data = readback_component.get_editor_property(
        name="animation_data"
    )
    readback_animation = readback_animation_data.get_editor_property(
        name="anim_to_play"
    )
    readback_mode = readback_component.get_animation_mode()
    if (
        _object_path(readback_mesh) != skeletal_mesh_path
        or _object_path(readback_animation) != walking_anim_path
        or readback_mode != unreal.AnimationMode.ANIMATION_SINGLE_NODE
    ):
        raise RuntimeError(
            "Blueprint readback must bind exactly the imported SkeletalMesh "
            "and Walking AnimSequence in single-node mode"
        )

    receipt = {
        "schema": IMPORT_JOB_RECEIPT_SCHEMA,
        "tag": TAG,
        "job_identity_sha256": job_identity_sha256,
        "skeletal_mesh": skeletal_mesh_path,
        "walking_animation": walking_anim_path,
        "blueprint": bp_path,
        "actions": EXPECTED_ACTIONS,
    }
    os.environ["GATE_IMPORT_RECEIPT_JSON"] = _canonical_json(receipt)

    spear.log(
        f"IMPORT_GATE_ANIMAL_OK tag={TAG} skm={skeletal_mesh_path} anim={walking_anim_path} bp={bp_path}"
    )


if __name__ == "__main__":
    main()
