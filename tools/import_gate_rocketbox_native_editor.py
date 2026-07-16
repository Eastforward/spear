"""Import one native Rocketbox runtime into an isolated Unreal gate."""

import hashlib
import json
import os
import posixpath
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

import spear
import unreal


TAG_CONTRACTS = {
    "rocketbox_female_adult_01_original_ue_v3": {
        "asset_id": "rocketbox_female_adult_01",
        "height_range_cm": [165.0, 200.0],
        "bottom_range_cm": [-5.0, 5.0],
        "relative_root": (
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_female_adult_01_original_ue_v3"
        ),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_native_ue_runtime_v3",
        "normalization_schema": (
            "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        ),
        "requires_in_place_walking": True,
        "ue_manifest_schema": "rocketbox_native_ue_import_v3",
        "ue_manifest_relative_path": (
            "tmp/rocketbox_native_ue_import_v3/"
            "rocketbox_female_adult_01_original_ue_v3/ue_import_manifest.json"
        ),
        "expected_material_names": [
            "f001_body",
            "f001_head",
            "f001_opacity",
        ],
        "expected_image_names": [
            "f001_body_specular",
            "f001_body_normal",
            "f001_body_color",
            "f001_head_specular",
            "f001_head_normal",
            "f001_head_color",
            "f001_opacity_color",
        ],
    },
    "rocketbox_male_adult_01_original_ue_v2": {
        "asset_id": "rocketbox_male_adult_01",
        "height_range_cm": [165.0, 200.0],
        "bottom_range_cm": [-5.0, 5.0],
        "relative_root": (
            "tmp/rocketbox_native_runtime_ue_v2/"
            "rocketbox_male_adult_01_original_ue_v2"
        ),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_native_ue_runtime_v2",
        "ue_manifest_relative_path": (
            "tmp/rocketbox_native_ue_import_v2/"
            "rocketbox_male_adult_01_original_ue_v2/ue_import_manifest.json"
        ),
    },
    "rocketbox_male_adult_01_shirt_blue_ue_v2": {
        "asset_id": "rocketbox_male_adult_01",
        "height_range_cm": [165.0, 200.0],
        "bottom_range_cm": [-5.0, 5.0],
        "relative_root": (
            "tmp/rocketbox_native_runtime_ue_v2/"
            "rocketbox_male_adult_01_shirt_blue_ue_v2"
        ),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_native_ue_runtime_v2",
        "ue_manifest_relative_path": (
            "tmp/rocketbox_native_ue_import_v2/"
            "rocketbox_male_adult_01_shirt_blue_ue_v2/ue_import_manifest.json"
        ),
    },
    "rocketbox_male_adult_01_original_ue_v3": {
        "asset_id": "rocketbox_male_adult_01",
        "height_range_cm": [165.0, 200.0],
        "bottom_range_cm": [-5.0, 5.0],
        "relative_root": (
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_male_adult_01_original_ue_v3"
        ),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_native_ue_runtime_v3",
        "normalization_schema": (
            "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        ),
        "requires_in_place_walking": True,
        "ue_manifest_schema": "rocketbox_native_ue_import_v3",
        "ue_manifest_relative_path": (
            "tmp/rocketbox_native_ue_import_v3/"
            "rocketbox_male_adult_01_original_ue_v3/ue_import_manifest.json"
        ),
    },
    "rocketbox_male_adult_01_shirt_blue_ue_v3": {
        "asset_id": "rocketbox_male_adult_01",
        "height_range_cm": [165.0, 200.0],
        "bottom_range_cm": [-5.0, 5.0],
        "relative_root": (
            "tmp/rocketbox_native_runtime_ue_v3/"
            "rocketbox_male_adult_01_shirt_blue_ue_v3"
        ),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_native_ue_runtime_v3",
        "normalization_schema": (
            "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        ),
        "requires_in_place_walking": True,
        "ue_manifest_schema": "rocketbox_native_ue_import_v3",
        "ue_manifest_relative_path": (
            "tmp/rocketbox_native_ue_import_v3/"
            "rocketbox_male_adult_01_shirt_blue_ue_v3/ue_import_manifest.json"
        ),
    },
}
USAGE_SCOPE = "research_candidate"
FORMAL_REGISTRATION_AUTHORIZED = False
UE_MANIFEST_SCHEMA = "rocketbox_native_ue_import_v2"
EXPECTED_BONE_COUNT = 80
EXPECTED_MATERIAL_NAMES = {"m002_body", "m002_head", "m002_opacity"}
EXPECTED_IMAGE_NAMES = {
    "m002_body_specular",
    "m002_body_normal",
    "m002_body_color",
    "m002_head_specular",
    "m002_head_normal",
    "m002_head_color",
    "m002_opacity_color",
}
REQUIRED_ANIMATION_NAMES = {"Walking", "Standing_Idle"}
REQUIRED_PRIMITIVE_ATTRIBUTES = {
    "POSITION",
    "TEXCOORD_0",
    "JOINTS_0",
    "WEIGHTS_0",
}


def _dynamic_batch_contract(tag):
    if os.environ.get("ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH") != "1":
        return None
    if re.fullmatch(r"rocketbox_[a-z0-9_]+_original_ue_v1", tag) is None:
        raise RuntimeError(f"unsafe dynamic Rocketbox batch tag: {tag!r}")
    root = Path(__file__).resolve().parents[1]
    normalized_root = Path(
        os.environ["ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT"]
    ).resolve()
    ue_manifest_root = Path(
        os.environ["ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT"]
    ).resolve()
    inventory_path = Path(os.environ["ROCKETBOX_NATIVE_INVENTORY_JSON"]).resolve()
    for path, description in (
        (normalized_root, "normalized root"),
        (ue_manifest_root, "UE manifest root"),
    ):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"dynamic Rocketbox {description} escaped SPEAR root: {path}"
            ) from error
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise RuntimeError("dynamic Rocketbox inventory is not a direct file")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise RuntimeError("dynamic Rocketbox inventory contract changed")
    base_avatar_id = tag[: -len("_original_ue_v1")]
    records = [
        record
        for record in inventory.get("avatars", [])
        if record.get("base_avatar_id") == base_avatar_id
    ]
    if len(records) != 1 or records[0].get("inventory_status") != "passed":
        raise RuntimeError("dynamic Rocketbox inventory avatar is not unique/ready")
    source_root = normalized_root / tag
    source_manifest_path = source_root / "normalization_manifest.json"
    source_glb = source_root / "runtime.glb"
    if (
        not source_manifest_path.is_file()
        or source_manifest_path.is_symlink()
        or not source_glb.is_file()
        or source_glb.is_symlink()
    ):
        raise RuntimeError("dynamic Rocketbox normalized source is incomplete")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    glb = source_manifest.get("glb_contract", {})
    expected_qa = source_manifest.get("expected_ue_qa", {})
    authored_bone_node_count = glb.get(
        "authored_bone_node_count",
        records[0].get("blender_audit", {}).get("bone_count"),
    )
    if (
        source_manifest.get("schema") != "rocketbox_batch_native_ue_runtime_v1"
        or source_manifest.get("tag") != tag
        or source_manifest.get("base_avatar_id") != base_avatar_id
        or source_manifest.get("usage_scope") != "research_candidate"
        or source_manifest.get("automatic_checks", {}).get("overall") != "passed"
        or expected_qa.get("actor_scale") != 1.0
        or glb.get("animation_names") != ["Walking", "Standing_Idle"]
        or authored_bone_node_count != 80
    ):
        raise RuntimeError("dynamic Rocketbox normalized manifest contract changed")
    material_names = glb.get("material_names")
    image_names = glb.get("image_names")
    if (
        not isinstance(material_names, list)
        or not material_names
        or len(material_names) != len(set(material_names))
        or not isinstance(image_names, list)
        or not image_names
        or len(image_names) != len(set(image_names))
    ):
        raise RuntimeError("dynamic Rocketbox material/image contract is invalid")
    ue_manifest = ue_manifest_root / tag / "ue_import_manifest.json"
    return {
        "asset_id": source_manifest["asset_id"],
        "base_avatar_id": base_avatar_id,
        "height_range_cm": expected_qa["height_range_cm"],
        "authored_height_cm": expected_qa["authored_height_cm"],
        "bottom_range_cm": expected_qa["bottom_range_cm"],
        "relative_root": source_root.relative_to(root).as_posix(),
        "runtime_glb": "runtime.glb",
        "source_manifest": "normalization_manifest.json",
        "source_manifest_schema": "rocketbox_batch_native_ue_runtime_v1",
        "normalization_schema": (
            "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        ),
        "requires_in_place_walking": True,
        "ue_manifest_schema": "rocketbox_batch_native_ue_import_v1",
        "ue_manifest_relative_path": ue_manifest.relative_to(root).as_posix(),
        "expected_material_names": material_names,
        "expected_image_names": image_names,
        "expected_primitive_count": int(glb["primitive_count"]),
        "expected_skin_joint_count": int(glb["joint_count"]),
        "expected_texture_count": len(image_names),
        "expected_skeleton_family": source_manifest["skeleton_family"],
    }


TAG = os.environ["ROCKETBOX_NATIVE_TAG"]
if TAG not in TAG_CONTRACTS:
    dynamic_contract = _dynamic_batch_contract(TAG)
    if dynamic_contract is not None:
        TAG_CONTRACTS[TAG] = dynamic_contract
ACTIVE_CONTRACT = TAG_CONTRACTS.get(TAG, {})
EXPECTED_MATERIAL_NAMES = set(
    ACTIVE_CONTRACT.get("expected_material_names", EXPECTED_MATERIAL_NAMES)
)
EXPECTED_IMAGE_NAMES = set(
    ACTIVE_CONTRACT.get("expected_image_names", EXPECTED_IMAGE_NAMES)
)
EXPECTED_PRIMITIVE_COUNT = int(
    ACTIVE_CONTRACT.get("expected_primitive_count", len(EXPECTED_MATERIAL_NAMES))
)
EXPECTED_SKIN_JOINT_COUNT = int(
    ACTIVE_CONTRACT.get("expected_skin_joint_count", EXPECTED_BONE_COUNT)
)
EXPECTED_TEXTURE_COUNT = int(
    ACTIVE_CONTRACT.get("expected_texture_count", len(EXPECTED_IMAGE_NAMES))
)
EXPECTED_SKELETON_FAMILY = ACTIVE_CONTRACT.get(
    "expected_skeleton_family", "Bip01"
)
SOURCE_GLB = Path(os.environ["ROCKETBOX_NATIVE_GLB"]).expanduser().resolve()
SOURCE_MANIFEST = Path(
    os.environ["ROCKETBOX_NATIVE_SOURCE_MANIFEST"]
).expanduser().resolve()
UE_MANIFEST = Path(
    os.environ["ROCKETBOX_NATIVE_UE_MANIFEST"]
).expanduser().resolve()
VERIFY_ONLY = os.environ.get("ROCKETBOX_NATIVE_VERIFY_ONLY") == "1"
MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, description):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must contain one JSON object")
    return value


def _write_json_atomic(path, payload, replace_existing=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_source_manifest(contract):
    manifest = _load_json(SOURCE_MANIFEST, "native Rocketbox source manifest")
    if manifest.get("schema") != contract["source_manifest_schema"]:
        raise RuntimeError("unexpected native Rocketbox source manifest schema")
    if manifest.get("tag") != TAG:
        raise RuntimeError("native Rocketbox source manifest tag mismatch")
    if manifest.get("asset_id") != contract["asset_id"]:
        raise RuntimeError("native Rocketbox source manifest asset mismatch")
    if manifest.get("usage_scope") != USAGE_SCOPE:
        raise RuntimeError("native Rocketbox source manifest scope mismatch")
    runtime = manifest.get("runtime_glb")
    if not isinstance(runtime, dict):
        raise RuntimeError("native Rocketbox source manifest lacks runtime_glb")
    if runtime.get("filename") != contract["runtime_glb"]:
        raise RuntimeError("native Rocketbox runtime filename mismatch")
    if runtime.get("sha256") != _sha256(SOURCE_GLB):
        raise RuntimeError("native Rocketbox runtime hash mismatch")
    if runtime.get("size_bytes") != SOURCE_GLB.stat().st_size:
        raise RuntimeError("native Rocketbox runtime size mismatch")
    normalization = manifest.get("normalization", {})
    expected_qa = manifest.get("expected_ue_qa", {})
    normalization_schema = contract.get(
        "normalization_schema",
        "rocketbox_ue_grounded_metric_skeleton_normalization_v1",
    )
    if (
        normalization.get("schema") != normalization_schema
        or normalization.get("normalized_joint_count") != EXPECTED_SKIN_JOINT_COUNT
        or normalization.get("static_wrapper_translation_zeroed") is not True
        or expected_qa.get("height_range_cm") != contract["height_range_cm"]
        or expected_qa.get("bottom_range_cm") != contract["bottom_range_cm"]
        or expected_qa.get("actor_scale") != 1.0
        or manifest.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise RuntimeError("native Rocketbox normalized UE contract changed")
    if contract.get("requires_in_place_walking", False):
        walking = normalization.get("root_motion", {}).get("Walking", {})
        runtime_motion = manifest.get("runtime_motion_contract", {})
        if (
            normalization.get("in_place_actions") != ["Walking"]
            or float(walking.get("maximum_horizontal_deviation_after_m", 1.0))
            >= 1.0e-6
            or float(walking.get("maximum_vertical_world_error_m", 1.0))
            >= 1.0e-6
            or runtime_motion.get("walking_embedded_horizontal_root_motion")
            != "removed"
            or runtime_motion.get("walking_vertical_motion") != "preserved"
            or runtime_motion.get("dynamic_ground_snap_to_floor_required")
            is not True
            or expected_qa.get("ground_snap_to_floor") is not True
        ):
            raise RuntimeError("native Rocketbox in-place Walking contract changed")
    return manifest


def _ue_manifest_schema(contract):
    return contract.get("ue_manifest_schema", UE_MANIFEST_SCHEMA)


def _validate_environment():
    if TAG not in TAG_CONTRACTS:
        raise RuntimeError(f"unsupported native Rocketbox tag: {TAG!r}")
    root = Path(__file__).resolve().parents[1]
    contract = TAG_CONTRACTS[TAG]
    expected_root = (root / contract["relative_root"]).resolve()
    expected_glb = (expected_root / contract["runtime_glb"]).resolve()
    expected_source_manifest = (
        expected_root / contract["source_manifest"]
    ).resolve()
    expected_ue_manifest = (
        root / contract["ue_manifest_relative_path"]
    ).resolve()
    if SOURCE_GLB != expected_glb or not SOURCE_GLB.is_file():
        raise RuntimeError(f"unexpected native Rocketbox runtime GLB: {SOURCE_GLB}")
    if SOURCE_MANIFEST != expected_source_manifest:
        raise RuntimeError(
            f"unexpected native Rocketbox source manifest: {SOURCE_MANIFEST}"
        )
    if UE_MANIFEST != expected_ue_manifest:
        raise RuntimeError(f"unexpected native Rocketbox UE manifest: {UE_MANIFEST}")
    source_manifest = _validate_source_manifest(contract)
    return contract, source_manifest


def _read_glb_contract(path):
    payload = Path(path).read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("native Rocketbox input is not GLB 2.0")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if (
        version != 2
        or declared_length != len(payload)
        or chunk_type != 0x4E4F534A
        or 20 + json_length > len(payload)
    ):
        raise RuntimeError("native Rocketbox GLB header is invalid")
    try:
        document = json.loads(payload[20 : 20 + json_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("native Rocketbox GLB JSON is invalid") from error

    meshes = document.get("meshes", [])
    if len(meshes) != 1:
        raise RuntimeError("native Rocketbox GLB must contain exactly one mesh")
    primitives = meshes[0].get("primitives", [])
    if len(primitives) != EXPECTED_PRIMITIVE_COUNT:
        raise RuntimeError("native Rocketbox GLB primitive count changed")
    if {primitive.get("material") for primitive in primitives} != set(
        range(len(EXPECTED_MATERIAL_NAMES))
    ):
        raise RuntimeError("native Rocketbox primitives do not use all materials")
    for primitive in primitives:
        attributes = primitive.get("attributes", {})
        if not REQUIRED_PRIMITIVE_ATTRIBUTES.issubset(attributes):
            raise RuntimeError("native Rocketbox primitive lacks UV or skin attributes")

    skins = document.get("skins", [])
    if len(skins) != 1:
        raise RuntimeError("native Rocketbox GLB must contain exactly one skin")
    joints = skins[0].get("joints", [])
    if len(joints) != EXPECTED_SKIN_JOINT_COUNT:
        raise RuntimeError("native Rocketbox GLB skin joint count changed")

    animations = document.get("animations", [])
    animation_name_values = [animation.get("name") for animation in animations]
    if any(not isinstance(name, str) for name in animation_name_values):
        raise RuntimeError("native Rocketbox action names are invalid")
    animation_names = set(animation_name_values)
    if len(animations) != 2 or animation_names != REQUIRED_ANIMATION_NAMES:
        raise RuntimeError(
            f"native Rocketbox actions differ: {sorted(animation_names)}"
        )

    materials = document.get("materials", [])
    material_name_values = [material.get("name") for material in materials]
    if any(not isinstance(name, str) for name in material_name_values):
        raise RuntimeError("native Rocketbox material names are invalid")
    material_names = set(material_name_values)
    if (
        len(materials) != len(EXPECTED_MATERIAL_NAMES)
        or material_names != EXPECTED_MATERIAL_NAMES
    ):
        raise RuntimeError(
            f"native Rocketbox materials differ: {sorted(material_names)}"
        )

    webp_extension = "EXT_texture_webp"
    if webp_extension in document.get("extensionsUsed", []):
        raise RuntimeError("native Rocketbox GLB must not use WebP")
    if webp_extension in document.get("extensionsRequired", []):
        raise RuntimeError("native Rocketbox GLB must not require WebP")
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or any("uri" in buffer for buffer in buffers):
        raise RuntimeError("native Rocketbox GLB must use one embedded buffer")

    images = document.get("images", [])
    image_name_values = [image.get("name") for image in images]
    if any(not isinstance(name, str) for name in image_name_values):
        raise RuntimeError("native Rocketbox image names are invalid")
    image_names = set(image_name_values)
    if (
        len(images) != len(EXPECTED_IMAGE_NAMES)
        or image_names != EXPECTED_IMAGE_NAMES
    ):
        raise RuntimeError(f"native Rocketbox images differ: {sorted(image_names)}")
    buffer_views = document.get("bufferViews", [])
    for image in images:
        image_extensions = image.get("extensions", {})
        buffer_view = image.get("bufferView")
        if (
            image.get("mimeType") != "image/png"
            or not isinstance(buffer_view, int)
            or not 0 <= buffer_view < len(buffer_views)
            or "uri" in image
            or not isinstance(image_extensions, dict)
            or webp_extension in image_extensions
        ):
            raise RuntimeError("native Rocketbox images must be embedded core PNG")

    textures = document.get("textures", [])
    if len(textures) != EXPECTED_TEXTURE_COUNT:
        raise RuntimeError("native Rocketbox GLB texture count changed")
    texture_sources = {texture.get("source") for texture in textures}
    if texture_sources != set(range(len(EXPECTED_IMAGE_NAMES))) or any(
        texture.get("extensions") for texture in textures
    ):
        raise RuntimeError("native Rocketbox textures must use core PNG sources")

    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    scene_index = document.get("scene", 0)
    if (
        not isinstance(nodes, list)
        or not isinstance(scenes, list)
        or not isinstance(scene_index, int)
        or not 0 <= scene_index < len(scenes)
        or not isinstance(scenes[scene_index], dict)
    ):
        raise RuntimeError("native Rocketbox normalized scene graph is invalid")
    roots = scenes[scene_index].get("nodes", [])
    mesh_node_indices = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and node.get("mesh") == 0 and node.get("skin") == 0
    ]
    armature_node_indices = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and node.get("name") == EXPECTED_SKELETON_FAMILY
    ]
    if len(mesh_node_indices) != 1 or mesh_node_indices[0] not in roots:
        raise RuntimeError("native Rocketbox skinned mesh must be a scene root")
    if len(armature_node_indices) != 1 or armature_node_indices[0] not in roots:
        raise RuntimeError("native Rocketbox armature wrapper must be a scene root")
    mesh_node_index = mesh_node_indices[0]
    if any(
        mesh_node_index in node.get("children", [])
        for node in nodes
        if isinstance(node, dict)
    ):
        raise RuntimeError("native Rocketbox skinned mesh still has a parent")
    armature_node = nodes[armature_node_indices[0]]
    if armature_node.get("scale") != [1.0, 1.0, 1.0]:
        raise RuntimeError("native Rocketbox armature unit scale is not normalized")
    if armature_node.get("translation") != [0.0, 0.0, 0.0]:
        raise RuntimeError("native Rocketbox armature wrapper is not grounded")
    if any(not isinstance(index, int) or not 0 <= index < len(nodes) for index in joints):
        raise RuntimeError("native Rocketbox skin contains an invalid joint index")

    return {
        "mesh_count": 1,
        "primitive_count": len(primitives),
        "skin_count": 1,
        "joint_count": len(joints),
        "animation_names": sorted(animation_names),
        "material_names": sorted(material_names),
        "image_names": sorted(image_names),
        "image_mime_types": sorted({image["mimeType"] for image in images}),
        "texture_count": len(textures),
        "mesh_is_scene_root": True,
        "armature_scale": list(armature_node["scale"]),
        "armature_translation": list(armature_node["translation"]),
    }


def _asset_record(asset_path):
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_name = str(
        data.get_editor_property("asset_class_path").get_editor_property(
            "asset_name"
        )
    )
    package_path = str(data.get_editor_property("package_path"))
    asset_name = str(data.get_editor_property("asset_name"))
    return {
        "asset_name": asset_name,
        "class_name": class_name,
        "object_path": posixpath.join(
            package_path, f"{asset_name}.{asset_name}"
        ),
    }


def _collect_imported_assets():
    assets = {
        "skeletal_mesh": [],
        "skeleton": [],
        "animations": {},
        "materials": [],
        "textures": [],
        "other": [],
    }
    paths = unreal.EditorAssetLibrary.list_assets(
        directory_path=MESH_DIR,
        recursive=True,
        include_folder=False,
    )
    for path in paths:
        record = _asset_record(path)
        class_name = record["class_name"]
        if class_name == "SkeletalMesh":
            assets["skeletal_mesh"].append(record)
        elif class_name == "Skeleton":
            assets["skeleton"].append(record)
        elif class_name == "AnimSequence":
            name = record["asset_name"]
            if name in assets["animations"]:
                raise RuntimeError(f"duplicate imported animation: {name}")
            assets["animations"][name] = record
        elif class_name in ("Material", "MaterialInstanceConstant"):
            assets["materials"].append(record)
        elif class_name == "Texture2D":
            assets["textures"].append(record)
        else:
            assets["other"].append(record)
    return assets


def _load_asset(record):
    asset = unreal.load_asset(name=record["object_path"])
    if asset is None:
        raise RuntimeError(f"could not load UE asset: {record['object_path']}")
    return asset


def _validate_runtime_assets(assets, component, contract):
    if len(assets["skeletal_mesh"]) != 1:
        raise RuntimeError("UE import must create exactly one SkeletalMesh")
    if len(assets["skeleton"]) != 1:
        raise RuntimeError("UE import must create exactly one Skeleton")
    if set(assets["animations"]) != REQUIRED_ANIMATION_NAMES:
        raise RuntimeError(
            f"UE imported wrong actions: {sorted(assets['animations'])}"
        )
    if len(assets["materials"]) != len(EXPECTED_MATERIAL_NAMES):
        raise RuntimeError("UE import material count changed")
    if len(assets["textures"]) != EXPECTED_TEXTURE_COUNT:
        raise RuntimeError("UE import texture count changed")

    mesh = _load_asset(assets["skeletal_mesh"][0])
    skeleton = _load_asset(assets["skeleton"][0])
    mesh_skeleton = mesh.get_editor_property("skeleton")
    if (
        mesh_skeleton is None
        or mesh_skeleton.get_path_name() != skeleton.get_path_name()
    ):
        raise RuntimeError("SkeletalMesh references the wrong imported Skeleton")
    for record in assets["animations"].values():
        animation_skeleton = _load_asset(record).get_editor_property("skeleton")
        if (
            animation_skeleton is None
            or animation_skeleton.get_path_name() != skeleton.get_path_name()
        ):
            raise RuntimeError(
                f"animation references the wrong Skeleton: {record['asset_name']}"
            )

    bone_count = component.get_num_bones()
    if bone_count != EXPECTED_BONE_COUNT:
        raise RuntimeError(
            f"UE imported {bone_count} bones, expected {EXPECTED_BONE_COUNT}"
        )
    slots = mesh.get_editor_property("materials")
    if len(slots) != len(EXPECTED_MATERIAL_NAMES):
        raise RuntimeError("SkeletalMesh material slot count changed")
    if any(slot.material_interface is None for slot in slots):
        raise RuntimeError("SkeletalMesh contains a null material slot")
    slot_names = {str(slot.material_slot_name) for slot in slots}
    if slot_names != EXPECTED_MATERIAL_NAMES:
        raise RuntimeError(f"SkeletalMesh material slots differ: {slot_names}")
    bounds = mesh.get_imported_bounds()
    height_cm = 2.0 * float(bounds.box_extent.z)
    bottom_cm = float(bounds.origin.z - bounds.box_extent.z)
    top_cm = float(bounds.origin.z + bounds.box_extent.z)
    height_range = contract["height_range_cm"]
    bottom_range = contract["bottom_range_cm"]
    if not height_range[0] <= height_cm <= height_range[1]:
        raise RuntimeError(
            f"SkeletalMesh height {height_cm} cm is outside {height_range}"
        )
    authored_height_cm = contract.get("authored_height_cm")
    authored_height_delta_cm = None
    authored_height_tolerance_cm = None
    if authored_height_cm is not None:
        authored_height_cm = float(authored_height_cm)
        authored_height_delta_cm = abs(height_cm - authored_height_cm)
        authored_height_tolerance_cm = max(3.0, authored_height_cm * 0.02)
        if authored_height_delta_cm > authored_height_tolerance_cm:
            raise RuntimeError(
                f"SkeletalMesh height changed from authored {authored_height_cm} cm "
                f"to {height_cm} cm"
            )
    if not bottom_range[0] <= bottom_cm <= bottom_range[1]:
        raise RuntimeError(
            f"SkeletalMesh bottom {bottom_cm} cm is outside {bottom_range}"
        )
    return {
        "bone_count": bone_count,
        "actor_scale": 1.0,
        "bounds": {
            "origin_cm": [
                float(bounds.origin.x),
                float(bounds.origin.y),
                float(bounds.origin.z),
            ],
            "box_extent_cm": [
                float(bounds.box_extent.x),
                float(bounds.box_extent.y),
                float(bounds.box_extent.z),
            ],
            "height_cm": height_cm,
            "bottom_cm": bottom_cm,
            "top_cm": top_cm,
            "height_range_cm": list(height_range),
            "bottom_range_cm": list(bottom_range),
            "height_passed": True,
            "authored_height_cm": authored_height_cm,
            "authored_height_delta_cm": authored_height_delta_cm,
            "authored_height_tolerance_cm": authored_height_tolerance_cm,
            "authored_height_preserved": (
                authored_height_cm is None
                or authored_height_delta_cm <= authored_height_tolerance_cm
            ),
            "ground_passed": True,
        },
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
        raise RuntimeError("native Rocketbox Blueprint lacks one skeletal component")
    component = components[0]
    component.set_animation_mode(
        animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE
    )
    component.set_skeletal_mesh_asset(
        new_mesh=_load_asset(assets["skeletal_mesh"][0])
    )
    component.set_editor_property(
        name="animation_data",
        value=unreal.SingleAnimationPlayData(
            anim_to_play=_load_asset(assets["animations"]["Walking"]),
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


def _content_record(assets, bp_path):
    return {
        "mesh_directory": MESH_DIR,
        "blueprint_directory": BP_DIR,
        "skeletal_mesh": assets["skeletal_mesh"][0]["object_path"],
        "skeleton": assets["skeleton"][0]["object_path"],
        "animations": {
            name: assets["animations"][name]["object_path"]
            for name in sorted(REQUIRED_ANIMATION_NAMES)
        },
        "materials": sorted(
            record["object_path"] for record in assets["materials"]
        ),
        "textures": sorted(record["object_path"] for record in assets["textures"]),
        "blueprint": bp_path,
    }


def _build_ue_manifest(
    contract,
    source_manifest,
    glb_contract,
    assets,
    runtime_contract,
    bp_path,
):
    return {
        "schema": _ue_manifest_schema(contract),
        "generated_at": _utc_now(),
        "tag": TAG,
        "asset_id": contract["asset_id"],
        "base_avatar_id": contract.get("base_avatar_id"),
        "usage_scope": USAGE_SCOPE,
        "formal_registration_authorized": FORMAL_REGISTRATION_AUTHORIZED,
        "source_glb": str(SOURCE_GLB),
        "source_glb_sha256": _sha256(SOURCE_GLB),
        "source_glb_size_bytes": SOURCE_GLB.stat().st_size,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "source_manifest_schema": source_manifest["schema"],
        "glb_contract": glb_contract,
        "content": _content_record(assets, bp_path),
        "runtime_contract": runtime_contract,
        "reload_verification": {"status": "pending"},
    }


def _cleanup_current_attempt(created_directories, manifest_created):
    if manifest_created and UE_MANIFEST.exists():
        UE_MANIFEST.unlink()
    for directory in reversed(created_directories):
        if unreal.EditorAssetLibrary.does_directory_exist(
            directory_path=directory
        ):
            unreal.EditorAssetLibrary.delete_directory(directory_path=directory)


def _blueprint_component(bp_path):
    blueprint = unreal.load_asset(name=bp_path)
    if blueprint is None:
        raise RuntimeError(f"could not reload native Rocketbox Blueprint: {bp_path}")
    subobjects = spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint
    )
    components = [
        item["object"]
        for item in subobjects
        if isinstance(item["object"], unreal.SkeletalMeshComponent)
    ]
    if len(components) != 1:
        raise RuntimeError("reloaded native Rocketbox Blueprint is invalid")
    return components[0]


def _verify_existing(contract, glb_contract):
    manifest = _load_json(UE_MANIFEST, "native Rocketbox UE import manifest")
    if manifest.get("schema") != _ue_manifest_schema(contract):
        raise RuntimeError("native Rocketbox UE manifest schema mismatch")
    if manifest.get("tag") != TAG:
        raise RuntimeError("native Rocketbox UE manifest tag mismatch")
    if manifest.get("asset_id") != contract["asset_id"]:
        raise RuntimeError("native Rocketbox UE manifest asset mismatch")
    if manifest.get("usage_scope") != USAGE_SCOPE:
        raise RuntimeError("native Rocketbox UE manifest scope mismatch")
    if manifest.get("formal_registration_authorized") is not False:
        raise RuntimeError("native Rocketbox UE manifest improperly authorizes registration")
    if Path(manifest.get("source_glb", "")).resolve() != SOURCE_GLB:
        raise RuntimeError("native Rocketbox UE manifest runtime path mismatch")
    if Path(manifest.get("source_manifest", "")).resolve() != SOURCE_MANIFEST:
        raise RuntimeError("native Rocketbox UE manifest source path mismatch")
    if manifest.get("source_glb_sha256") != _sha256(SOURCE_GLB):
        raise RuntimeError("native Rocketbox runtime changed after UE import")
    if manifest.get("source_manifest_sha256") != _sha256(SOURCE_MANIFEST):
        raise RuntimeError("native Rocketbox source manifest changed after UE import")
    if manifest.get("glb_contract") != glb_contract:
        raise RuntimeError("native Rocketbox GLB contract changed after UE import")
    content = manifest.get("content", {})
    if (
        content.get("mesh_directory") != MESH_DIR
        or content.get("blueprint_directory") != BP_DIR
    ):
        raise RuntimeError("native Rocketbox UE content escaped its isolated gate")
    assets = _collect_imported_assets()
    component = _blueprint_component(content.get("blueprint"))
    runtime = _validate_runtime_assets(assets, component, contract)
    manifest["runtime_contract"] = runtime
    manifest["reload_verification"] = {
        "status": "passed",
        "verified_at": _utc_now(),
        "process": "second_ue_commandlet",
    }
    _write_json_atomic(UE_MANIFEST, manifest)
    spear.log(
        f"VERIFY_ROCKETBOX_NATIVE_IMPORT_OK tag={TAG} manifest={UE_MANIFEST}"
    )


def main():
    contract, source_manifest = _validate_environment()
    glb_contract = _read_glb_contract(SOURCE_GLB)
    if VERIFY_ONLY:
        _verify_existing(contract, glb_contract)
        return

    if UE_MANIFEST.exists():
        raise RuntimeError(
            f"refusing to replace existing UE manifest: {UE_MANIFEST}"
        )
    for directory in (MESH_DIR, BP_DIR):
        if unreal.EditorAssetLibrary.does_directory_exist(
            directory_path=directory
        ):
            raise RuntimeError(
                f"refusing to replace existing UE directory: {directory}"
            )

    created_directories = []
    manifest_created = False
    try:
        for directory in (MESH_DIR, BP_DIR):
            if not unreal.EditorAssetLibrary.make_directory(
                directory_path=directory
            ):
                raise RuntimeError(
                    f"could not create isolated UE directory: {directory}"
                )
            created_directories.append(directory)

        task = unreal.AssetImportTask()
        task.set_editor_property(name="async_", value=True)
        task.set_editor_property(name="automated", value=True)
        task.set_editor_property(name="destination_path", value=MESH_DIR)
        task.set_editor_property(name="filename", value=str(SOURCE_GLB))
        task.set_editor_property(name="replace_existing", value=False)
        task.set_editor_property(
            name="replace_existing_settings", value=False
        )
        task.set_editor_property(name="save", value=False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(
            import_tasks=[task]
        )
        imported_objects = task.get_objects()
        if not imported_objects:
            raise RuntimeError(
                "Interchange completed without native Rocketbox objects"
            )
        unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
        unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
            directory_path=MESH_DIR,
            only_if_is_dirty=False,
            recursive=True,
        )

        assets = _collect_imported_assets()
        bp_path, component = _create_blueprint(assets)
        runtime_contract = _validate_runtime_assets(assets, component, contract)
        manifest = _build_ue_manifest(
            contract,
            source_manifest,
            glb_contract,
            assets,
            runtime_contract,
            bp_path,
        )
        _write_json_atomic(UE_MANIFEST, manifest, replace_existing=False)
        manifest_created = True
    except BaseException:
        _cleanup_current_attempt(created_directories, manifest_created)
        raise

    spear.log(
        f"IMPORT_ROCKETBOX_NATIVE_OK tag={TAG} "
        f"bp={bp_path} manifest={UE_MANIFEST}"
    )


if __name__ == "__main__":
    main()
