"""Static contracts for the isolated native Rocketbox UE importer."""

import ast
import copy
import json
import struct

import pytest

from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "import_gate_rocketbox_native_editor.py"
)


def test_native_rocketbox_importer_exists_as_an_isolated_tool():
    assert SCRIPT_PATH.is_file()


def _source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _tree():
    return ast.parse(_source())


def _function_source(name):
    source = _source()
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def _literal_assignment(name):
    for node in _tree().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def test_importer_accepts_pinned_v2_evidence_and_new_in_place_v3_tags():
    contracts = _literal_assignment("TAG_CONTRACTS")
    assert sorted(contracts) == [
        "rocketbox_female_adult_01_original_ue_v3",
        "rocketbox_male_adult_01_original_ue_v2",
        "rocketbox_male_adult_01_original_ue_v3",
        "rocketbox_male_adult_01_shirt_blue_ue_v2",
        "rocketbox_male_adult_01_shirt_blue_ue_v3",
    ]
    for tag, contract in contracts.items():
        assert contract["asset_id"] in {
            "rocketbox_male_adult_01",
            "rocketbox_female_adult_01",
        }
        assert contract["height_range_cm"] == [165.0, 200.0]
        assert contract["bottom_range_cm"] == [-5.0, 5.0]
        assert contract["runtime_glb"] == "runtime.glb"
        assert contract["source_manifest"] == "normalization_manifest.json"
        assert tag in contract["relative_root"]
        assert tag in contract["ue_manifest_relative_path"]
        if tag.endswith("_ue_v3"):
            assert contract["source_manifest_schema"] == (
                "rocketbox_native_ue_runtime_v3"
            )
            assert contract["normalization_schema"] == (
                "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
            )
            assert contract["ue_manifest_schema"] == (
                "rocketbox_native_ue_import_v3"
            )
            assert contract["requires_in_place_walking"] is True
        else:
            assert contract["source_manifest_schema"] == (
                "rocketbox_native_ue_runtime_v2"
            )
    female = contracts["rocketbox_female_adult_01_original_ue_v3"]
    assert female["expected_material_names"] == [
        "f001_body",
        "f001_head",
        "f001_opacity",
    ]
    assert len(female["expected_image_names"]) == 7

    validate = _function_source("_validate_environment")
    assert "TAG not in TAG_CONTRACTS" in validate
    assert 'root / contract["relative_root"]' in validate
    assert 'contract["runtime_glb"]' in validate
    assert 'contract["source_manifest"]' in validate
    assert 'root / contract["ue_manifest_relative_path"]' in validate
    assert "SOURCE_GLB != expected_glb" in validate
    assert "SOURCE_MANIFEST != expected_source_manifest" in validate
    assert "UE_MANIFEST != expected_ue_manifest" in validate


def test_importer_supports_fail_closed_inventory_bound_batch_contracts():
    source = _source()
    assert "ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH" in source
    assert "ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT" in source
    assert "ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT" in source
    assert "ROCKETBOX_NATIVE_INVENTORY_JSON" in source
    dynamic = _function_source("_dynamic_batch_contract")
    assert "rocketbox_batch_native_ue_runtime_v1" in dynamic
    assert "rocketbox_human_inventory_v1" in dynamic
    assert "base_avatar_id" in dynamic
    assert "expected_material_names" in dynamic
    assert "expected_image_names" in dynamic


def test_source_manifest_authenticates_runtime_identity_hash_and_scope():
    validate = _function_source("_validate_source_manifest")

    assert 'manifest.get("schema") != contract["source_manifest_schema"]' in validate
    assert 'manifest.get("tag") != TAG' in validate
    assert 'manifest.get("asset_id") != contract["asset_id"]' in validate
    assert 'manifest.get("usage_scope") != USAGE_SCOPE' in validate
    assert 'runtime.get("filename") != contract["runtime_glb"]' in validate
    assert 'runtime.get("sha256") != _sha256(SOURCE_GLB)' in validate
    assert 'runtime.get("size_bytes") != SOURCE_GLB.stat().st_size' in validate
    assert '"normalization_schema"' in validate
    assert 'contract.get("requires_in_place_walking", False)' in validate
    assert 'runtime_motion.get("walking_embedded_horizontal_root_motion")' in validate


def test_native_importer_does_not_reuse_hunyuan_gate_or_scope():
    source = _source().lower()

    assert "human_apartment_gate" not in source
    assert "technical_spike_only" not in source
    assert "hy3d" not in source
    assert _literal_assignment("USAGE_SCOPE") == "research_candidate"
    assert _literal_assignment("FORMAL_REGISTRATION_AUTHORIZED") is False


EXPECTED_IMAGES = {
    "m002_body_specular",
    "m002_body_normal",
    "m002_body_color",
    "m002_head_specular",
    "m002_head_normal",
    "m002_head_color",
    "m002_opacity_color",
}


def _valid_glb_document():
    names = sorted(EXPECTED_IMAGES)
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [81, 80]}],
        "nodes": [
            {"name": f"Joint_{index}"} for index in range(80)
        ]
        + [
            {"name": "Body", "mesh": 0, "skin": 0},
            {
                "name": "Bip01",
                "children": [0],
                "scale": [1.0, 1.0, 1.0],
                "translation": [0.0, 0.0, 0.0],
            },
        ],
        "buffers": [{"byteLength": 0}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 0}
            for _ in names
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "TEXCOORD_0": 1,
                            "JOINTS_0": 2,
                            "WEIGHTS_0": 3,
                        },
                        "indices": 4,
                        "material": index,
                    }
                    for index in range(3)
                ]
            }
        ],
        "skins": [{"joints": list(range(80))}],
        "animations": [{"name": "Walking"}, {"name": "Standing_Idle"}],
        "materials": [
            {"name": "m002_body"},
            {"name": "m002_head"},
            {"name": "m002_opacity"},
        ],
        "images": [
            {"name": name, "mimeType": "image/png", "bufferView": index}
            for index, name in enumerate(names)
        ],
        "textures": [{"source": index} for index in range(len(names))],
    }


def _write_json_only_glb(path, document):
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )
    return path


def _load_preflight():
    namespace = {
        "Path": Path,
        "json": json,
        "struct": struct,
        "EXPECTED_BONE_COUNT": 80,
        "EXPECTED_SKIN_JOINT_COUNT": 80,
        "EXPECTED_PRIMITIVE_COUNT": 3,
        "EXPECTED_TEXTURE_COUNT": 7,
        "EXPECTED_SKELETON_FAMILY": "Bip01",
        "EXPECTED_MATERIAL_NAMES": {
            "m002_body",
            "m002_head",
            "m002_opacity",
        },
        "EXPECTED_IMAGE_NAMES": EXPECTED_IMAGES,
        "REQUIRED_ANIMATION_NAMES": {"Walking", "Standing_Idle"},
        "REQUIRED_PRIMITIVE_ATTRIBUTES": {
            "POSITION",
            "TEXCOORD_0",
            "JOINTS_0",
            "WEIGHTS_0",
        },
    }
    exec(_function_source("_read_glb_contract"), namespace)
    return namespace["_read_glb_contract"]


def test_glb_preflight_accepts_only_the_complete_native_runtime_contract(tmp_path):
    preflight = _load_preflight()
    path = _write_json_only_glb(tmp_path / "runtime.glb", _valid_glb_document())

    assert preflight(path) == {
        "mesh_count": 1,
        "primitive_count": 3,
        "skin_count": 1,
        "joint_count": 80,
        "animation_names": ["Standing_Idle", "Walking"],
        "material_names": ["m002_body", "m002_head", "m002_opacity"],
        "image_names": sorted(EXPECTED_IMAGES),
        "image_mime_types": ["image/png"],
        "texture_count": 7,
        "mesh_is_scene_root": True,
        "armature_scale": [1.0, 1.0, 1.0],
        "armature_translation": [0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["meshes"].append(copy.deepcopy(document["meshes"][0])),
        lambda document: document["meshes"][0]["primitives"].pop(),
        lambda document: document["skins"].append(copy.deepcopy(document["skins"][0])),
        lambda document: document["skins"][0]["joints"].pop(),
        lambda document: document["animations"].append({"name": "Walking"}),
        lambda document: document["animations"].__setitem__(0, {"name": "Running"}),
        lambda document: document["animations"][0].__setitem__("name", None),
        lambda document: document["materials"].__setitem__(0, {"name": "Material_0"}),
        lambda document: document["materials"][0].__setitem__("name", None),
        lambda document: document["images"].pop(),
        lambda document: document["images"][0].__setitem__("name", None),
        lambda document: document["images"][0].__setitem__("mimeType", "image/webp"),
        lambda document: document["images"][0].__setitem__("bufferView", 99),
        lambda document: document["images"][0].__setitem__("extensions", None),
        lambda document: document["images"][0].__setitem__("uri", "body.png"),
        lambda document: document["textures"][0].__setitem__(
            "extensions", {"EXT_texture_webp": {"source": 0}}
        ),
        lambda document: document.__setitem__(
            "extensionsRequired", ["EXT_texture_webp"]
        ),
        lambda document: document["buffers"][0].__setitem__("uri", "runtime.bin"),
        lambda document: document["nodes"][81].__setitem__(
            "scale", [0.01, 0.01, 0.01]
        ),
        lambda document: document["nodes"][81].__setitem__(
            "translation", [0.0, 0.895, 0.0]
        ),
        lambda document: document["scenes"][0].__setitem__("nodes", [81]),
    ],
)
def test_glb_preflight_rejects_wrong_structure_webp_and_external_uris(
    tmp_path, mutation
):
    preflight = _load_preflight()
    document = _valid_glb_document()
    mutation(document)
    path = _write_json_only_glb(tmp_path / "runtime.glb", document)

    with pytest.raises(RuntimeError):
        preflight(path)


def test_runtime_validation_requires_exact_native_assets_bones_and_slots():
    collect = _function_source("_collect_imported_assets")
    validate = _function_source("_validate_runtime_assets")

    for class_name in (
        "SkeletalMesh",
        "Skeleton",
        "AnimSequence",
        "Material",
        "MaterialInstanceConstant",
        "Texture2D",
    ):
        assert class_name in collect
    assert 'len(assets["skeletal_mesh"]) != 1' in validate
    assert 'len(assets["skeleton"]) != 1' in validate
    assert 'set(assets["animations"]) != REQUIRED_ANIMATION_NAMES' in validate
    assert 'len(assets["materials"]) != len(EXPECTED_MATERIAL_NAMES)' in validate
    assert 'len(assets["textures"]) != EXPECTED_TEXTURE_COUNT' in validate
    assert "component.get_num_bones()" in validate
    assert "bone_count != EXPECTED_BONE_COUNT" in validate
    assert "len(slots) != len(EXPECTED_MATERIAL_NAMES)" in validate
    assert "slot.material_interface is None" in validate
    assert "slot_names != EXPECTED_MATERIAL_NAMES" in validate
    assert 'get_editor_property("skeleton")' in validate
    assert "mesh.get_imported_bounds()" in validate
    assert 'contract["height_range_cm"]' in validate
    assert 'contract.get("authored_height_cm")' in validate
    assert '"authored_height_delta_cm"' in validate
    assert 'contract["bottom_range_cm"]' in validate
    assert '"height_cm"' in validate
    assert '"bottom_cm"' in validate
    assert '"actor_scale": 1.0' in validate


def test_blueprint_defaults_to_walking_and_always_ticks_the_pose():
    create = _function_source("_create_blueprint")

    assert "unreal.SkeletalMeshActor" in create
    assert "unreal.AnimationMode.ANIMATION_SINGLE_NODE" in create
    assert 'assets["animations"]["Walking"]' in create
    assert "unreal.SingleAnimationPlayData" in create
    assert "ALWAYS_TICK_POSE_AND_REFRESH_BONES" in create
    assert "save_loaded_asset" in create


def test_ue_manifest_is_atomic_hash_locked_and_not_formally_registered():
    atomic = _function_source("_write_json_atomic")
    build = _function_source("_build_ue_manifest")

    assert "os.O_EXCL" in atomic
    assert "os.replace" in atomic
    assert '"schema": _ue_manifest_schema(contract)' in build
    assert '"usage_scope": USAGE_SCOPE' in build
    assert '"formal_registration_authorized": FORMAL_REGISTRATION_AUTHORIZED' in build
    assert '"source_glb_sha256": _sha256(SOURCE_GLB)' in build
    assert '"source_manifest_sha256": _sha256(SOURCE_MANIFEST)' in build
    assert '"reload_verification": {"status": "pending"}' in build
    assert _literal_assignment("UE_MANIFEST_SCHEMA") == (
        "rocketbox_native_ue_import_v2"
    )


def test_import_is_strictly_no_replace_and_waits_for_interchange():
    main = _function_source("main")

    assert "if UE_MANIFEST.exists():" in main
    assert "refusing to replace existing UE manifest" in main
    assert "for directory in (MESH_DIR, BP_DIR):" in main
    assert "does_directory_exist" in main
    assert "refusing to replace existing UE directory" in main
    assert "created_directories = []" in main
    assert "created_directories.append(directory)" in main
    assert 'name="replace_existing", value=False' in main
    assert 'name="replace_existing_settings", value=False' in main
    assert "import_asset_tasks" in main
    assert "task.get_objects()" in main
    assert "wait_for_completion()" in main
    assert "save_directory" in main
    assert (
        "_write_json_atomic(UE_MANIFEST, manifest, replace_existing=False)"
        in main
    )
    assert main.index("_validate_runtime_assets") < main.index("_write_json_atomic")


def test_failure_cleanup_only_deletes_directories_created_by_this_attempt():
    cleanup = _function_source("_cleanup_current_attempt")
    main = _function_source("main")

    assert "created_directories" in cleanup
    assert "reversed(created_directories)" in cleanup
    assert "delete_directory" in cleanup
    assert "manifest_created" in cleanup
    assert "if manifest_created and UE_MANIFEST.exists():" in cleanup
    assert "for directory in (MESH_DIR, BP_DIR)" not in cleanup
    assert "except BaseException:" in main
    assert "_cleanup_current_attempt(created_directories, manifest_created)" in main


def test_second_commandlet_reloads_and_revalidates_the_exact_import():
    verify = _function_source("_verify_existing")
    main = _function_source("main")

    assert 'manifest.get("schema") != _ue_manifest_schema(contract)' in verify
    assert 'manifest.get("tag") != TAG' in verify
    assert 'manifest.get("usage_scope") != USAGE_SCOPE' in verify
    assert 'manifest.get("formal_registration_authorized") is not False' in verify
    assert 'Path(manifest.get("source_glb", "")).resolve() != SOURCE_GLB' in verify
    assert (
        'Path(manifest.get("source_manifest", "")).resolve() != SOURCE_MANIFEST'
        in verify
    )
    assert 'manifest.get("source_glb_sha256") != _sha256(SOURCE_GLB)' in verify
    assert 'manifest.get("source_manifest_sha256") != _sha256(SOURCE_MANIFEST)' in verify
    assert "_collect_imported_assets()" in verify
    assert "_validate_runtime_assets" in verify
    assert '"status": "passed"' in verify
    assert '"process": "second_ue_commandlet"' in verify
    assert "_write_json_atomic(UE_MANIFEST, manifest)" in verify
    assert "if VERIFY_ONLY:" in main
    assert "_verify_existing(contract, glb_contract)" in main
