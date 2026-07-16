"""Static contracts for the isolated Rocketbox/Hunyuan UE importer."""

import ast
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "import_gate_humanoid_editor.py"
)


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


def test_importer_is_confined_to_the_two_reviewed_spike_tags_and_runtime_glbs():
    assert _literal_assignment("ALLOWED_TAG_TO_ASSET_ID") == {
        "hy3d_rocketbox_male_adult_01_spike": "rocketbox_male_adult_01",
        "hy3d_rocketbox_female_adult_01_spike": "rocketbox_female_adult_01",
    }
    validate = _function_source("_validate_environment")

    assert "Path(__file__).resolve().parents[1]" in validate
    assert '"tmp" / "hy3d_rocketbox_template_fit_v1"' in validate
    assert 'asset_id / "ue_runtime.glb"' in validate
    assert "source_glb != expected_glb" in validate
    assert 'asset_id / "ue_import_manifest.json"' in validate
    assert "manifest_path != expected_manifest" in validate


def test_glb_preflight_requires_one_mesh_one_skin_80_joints_and_two_actions():
    preflight = _function_source("_read_glb_contract")

    assert 'payload[:4] != b"glTF"' in preflight
    assert 'len(document.get("meshes", [])) != 1' in preflight
    assert 'len(document.get("skins", [])) != 1' in preflight
    assert 'len(skin.get("joints", [])) != EXPECTED_BONE_COUNT' in preflight
    assert 'animation_names != REQUIRED_ANIMATION_NAMES' in preflight
    assert 'len(document.get("materials", [])) < EXPECTED_MATERIAL_SLOTS' in preflight


def test_import_requires_exact_asset_classes_bones_materials_and_textures():
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
    assert 'set(assets["animations"]) != set(REQUIRED_ANIMATION_NAMES)' in validate
    assert 'len(assets["materials"]) < EXPECTED_MATERIAL_SLOTS' in validate
    assert 'len(assets["textures"]) < EXPECTED_MATERIAL_SLOTS' in validate
    assert "smc.get_num_bones()" in validate
    assert "bone_count != EXPECTED_BONE_COUNT" in validate
    assert "material_interface is None" in validate


def test_blueprint_defaults_to_walking_and_keeps_scene_capture_animation_ticking():
    create = _function_source("_create_blueprint")

    assert "unreal.SkeletalMeshActor" in create
    assert "unreal.AnimationMode.ANIMATION_SINGLE_NODE" in create
    assert 'animations["Walking"]' in create
    assert "unreal.SingleAnimationPlayData" in create
    assert "ALWAYS_TICK_POSE_AND_REFRESH_BONES" in create


def test_manifest_is_technical_spike_only_atomic_and_reload_verified():
    source = _source()
    atomic = _function_source("_atomic_write_json")
    manifest = _function_source("_build_manifest")
    verify = _function_source("_verify_only")

    assert _literal_assignment("USAGE_SCOPE") == "technical_spike_only"
    assert _literal_assignment("MANIFEST_SCHEMA") == "hy3d_rocketbox_ue_import_v1"
    assert "os.O_EXCL" in atomic
    assert "os.replace" in atomic
    assert '"source_glb_sha256"' in manifest
    assert '"reload_verification"' in manifest
    assert '"status": "pending"' in manifest
    assert '"status"] = "passed"' in verify
    assert "_atomic_write_json" in verify
    assert "GATE_VERIFY_ONLY" in source


def test_partial_import_failure_removes_both_content_dirs_and_no_manifest_is_published():
    cleanup = _function_source("_cleanup_partial_import")
    main = _function_source("main")

    assert "MESH_CONTENT_DIR" in cleanup
    assert "BP_CONTENT_DIR" in cleanup
    assert "delete_directory" in cleanup
    assert "manifest_path.unlink" in cleanup
    assert "except BaseException:" in main
    assert "_cleanup_partial_import" in main
    assert main.index("_validate_runtime_assets") < main.index("_atomic_write_json")
