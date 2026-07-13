"""Static contract for the sealed Rocketbox native Walk/Idle runtime builder."""

import ast
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_build_native_rocketbox_runtime.py"


def script_source():
    assert SCRIPT.is_file(), f"missing native Rocketbox runtime builder: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def compact_source():
    return "".join(script_source().split())


def function_source(name):
    source = script_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def module_constant(name):
    tree = ast.parse(script_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def test_builder_pins_only_the_sealed_male_inputs_and_native_output():
    source = script_source()

    assert module_constant("ASSET_ID") == "rocketbox_male_adult_01"
    assert module_constant("BASELINE_MANIFEST_SHA256") == (
        "b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922"
    )
    assert module_constant("BASELINE_BLEND_SHA256") == (
        "951859fec42091e2e71cc99536996bd29a5536b6e8262f0576fb2b3459fbe603"
    )
    assert module_constant("ROCKETBOX_COMMIT") == (
        "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
    )
    assert module_constant("IDLE_SHA256") == (
        "818cc185af21390575f7fbfdeb3012ba2ce5969fbcb220ea725a2617b339a6e2"
    )
    assert module_constant("IDLE_GIT_BLOB_SHA1") == (
        "a2d92c3326a9c503af677c9fa6082387f060d6c4"
    )
    assert "/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1" in source
    assert "/data/datasets/rocketbox/Microsoft-Rocketbox" in source
    assert "Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx" in source
    assert (
        "tmp/rocketbox_native_runtime_v1/rocketbox_male_adult_01_original_v1"
        in source
    )
    assert "rocketbox_male_adult_01_shirt_blue_v1" in source
    assert "build_manifest.json" in source
    assert "variant_manifest.json" in source


def test_optional_body_color_variant_requires_a_paired_manifest():
    parse = function_source("parse_args")
    variant = function_source("validate_variant_request")

    assert "--body-color-texture" in parse
    assert "--variant-manifest" in parse
    assert "body_color_texture" in variant
    assert "variant_manifest" in variant
    assert "rocketbox_native_body_color_variant_v1" in variant
    assert "m002_body_color" in variant
    assert "variant_id" in variant
    assert "sha256" in variant


def test_builder_authenticates_clone_commit_blob_and_all_sealed_bytes():
    validate = function_source("validate_inputs")

    for evidence in (
        "baseline_manifest.json",
        "rocketbox_baseline_manifest_v1",
        "rocketbox_neutral_walk_v1",
        "retarget.blend",
        "BASELINE_MANIFEST_SHA256",
        "BASELINE_BLEND_SHA256",
        "ROCKETBOX_COMMIT",
        "IDLE_SHA256",
        "IDLE_GIT_BLOB_SHA1",
        "git",
        "rev-parse",
        "hash-object",
    ):
        assert evidence in validate
    assert "TEXTURE_DIR.relative_to(ROCKETBOX_ROOT)" in validate


def test_builder_opens_only_a_verified_staging_copy_and_never_saves_the_baseline():
    stage = function_source("stage_inputs")
    build = function_source("build_runtime")
    source = script_source()

    assert "shutil.copyfile" in stage or "copy_authenticated_file" in stage
    assert "sha256_file" in stage
    assert "staged_blend" in build
    assert "bpy.ops.wm.open_mainfile" in build
    assert "BASELINE_BLEND" not in build.split("open_mainfile", 1)[1].split(")", 1)[0]
    assert "save_as_mainfile" not in source
    assert "save_mainfile" not in source
    assert "os.replace" in function_source("publish_staging")
    assert "exists()" in function_source("publish_staging")


def test_builder_preserves_the_native_mesh_skin_material_and_image_contract():
    capture = function_source("capture_native_contract")
    compare = function_source("assert_native_contract_unchanged")

    for evidence in (
        "mesh_metrics",
        "capture_skin_contract",
        "80",
        "material_slot_names",
        "image_payload",
        "rest",
        "parent",
    ):
        assert evidence in capture or evidence in compare
    assert "3" in compare
    assert "7" in compare
    assert "bind_mesh_sha256" in compare


def test_variant_replaces_only_the_packed_body_color_and_compares_to_original():
    replace = function_source("replace_body_color_texture")
    compare = function_source("assert_variant_matches_original")
    build = function_source("build_runtime")

    assert "m002_body_color" in replace
    assert ".pack(" in replace
    assert "TEX_IMAGE" in replace
    assert "m002_body_color" in compare
    assert "other_image_payload_sha256" in compare
    assert "mesh_skin_action_contract_sha256" in compare
    assert "ORIGINAL_OUTPUT_DIR" in compare
    assert "replace_body_color_texture" in build
    assert "assert_variant_matches_original" in build


def test_builder_reuses_the_official_idle_bake_on_the_existing_armature():
    build = function_source("build_runtime")

    assert "identify_target_objects" in build
    assert "bake_idle_action" in build
    assert "remove_original_body" not in build
    assert "import_hy3d_obj" not in build
    assert "bind_target_mesh" not in build
    assert "validate_two_actions" in build
    assert "Walking" in build
    assert "Standing_Idle" in build


def test_combined_export_is_selection_only_and_contains_exact_named_nla_actions():
    export = function_source("export_combined_glb")

    for evidence in (
        "NLA_TRACKS",
        "Walking",
        "Standing_Idle",
        "use_selection=True",
        "export_animations=True",
        "export_force_sampling=True",
        "export_skins=True",
        "export_texcoords=True",
        "export_normals=True",
    ):
        assert evidence in export


def test_glb_contract_and_both_action_roundtrips_are_manifested():
    inspect = function_source("inspect_combined_glb")
    roundtrip = function_source("roundtrip_validate_combined")
    build = function_source("build_runtime")

    for evidence in (
        "mesh_count",
        "skin_count",
        "skin_joint_count",
        "animation_names",
        "material_count",
        "image_count",
        "Walking",
        "Standing_Idle",
    ):
        assert evidence in inspect
    assert "roundtrip_validate" in roundtrip
    assert "Walking" in roundtrip
    assert "Standing_Idle" in roundtrip
    assert "build_manifest.json" in script_source()
    assert "variant_manifest.json" in script_source()
    assert "source_hashes_before" in build
    assert "source_hashes_after" in build
    assert "glb_contract" in build
    assert "glb_roundtrip" in build
    assert "formal_dataset_asset" in build


def test_builder_emits_the_exact_ue_importer_source_manifest_contract():
    identity = function_source("output_identity")
    build = function_source("build_runtime")

    assert "rocketbox_native_runtime_build_v1" in identity
    assert "rocketbox_native_material_variant_v1" in identity
    assert "ORIGINAL_TAG" in identity
    assert "build_manifest.json" in identity
    assert "variant_manifest.json" in identity
    for field in ('"schema"', '"tag"', '"asset_id"', '"usage_scope"', '"runtime_glb"'):
        assert field in build
    assert '"usage_scope": "research_candidate"' in build


def test_source_hashes_before_are_captured_before_blender_build_starts():
    main = function_source("main")
    build = function_source("build_runtime")

    assert main.index("collect_source_hashes") < main.index("build_runtime")
    assert "source_hashes_before" in build
    assert "source_hashes_after = collect_source_hashes" in build


def test_roundtrip_selects_the_single_skinned_runtime_mesh_not_import_helpers():
    identify = function_source("identify_runtime_objects")
    roundtrip = function_source("roundtrip_validate_combined")

    assert 'obj.type == "ARMATURE"' in identify
    assert 'obj.type == "MESH"' in identify
    assert 'modifier.type == "ARMATURE"' in identify
    assert "modifier.object is armature" in identify
    assert "skinned_meshes" in identify
    assert "identify_runtime_objects" in roundtrip
    assert "direct.identify_target_objects" not in roundtrip


def test_roundtrip_authenticates_exact_nla_names_before_normalizing_blender_names():
    namespace = {"EXPECTED_ACTIONS": ("Walking", "Standing_Idle")}
    exec(function_source("normalize_imported_actions"), namespace)
    normalize = namespace["normalize_imported_actions"]

    walking = SimpleNamespace(name="Walking_Bip01")
    idle = SimpleNamespace(name="Standing_Idle_Bip01")
    armature = SimpleNamespace(
        animation_data=SimpleNamespace(
            action=walking,
            nla_tracks=[
                SimpleNamespace(
                    name="Walking",
                    strips=[SimpleNamespace(action=walking)],
                ),
                SimpleNamespace(
                    name="Standing_Idle",
                    strips=[SimpleNamespace(action=idle)],
                ),
            ],
        )
    )

    actions, evidence = normalize(armature, [walking, idle])

    assert set(actions) == {"Walking", "Standing_Idle"}
    assert walking.name == "Walking"
    assert idle.name == "Standing_Idle"
    assert evidence["Walking"]["imported_action_datablock_name"] == "Walking_Bip01"
    assert evidence["Standing_Idle"]["nla_track_name"] == "Standing_Idle"


def test_builder_has_fail_closed_no_replace_cleanup_and_success_sentinel():
    main = function_source("main")
    source = script_source()

    assert "try" in main and "finally" in main
    assert "cleanup" in main
    assert "ROCKETBOX_NATIVE_RUNTIME_OK" in source
    assert "formal_dataset_asset" in source
    assert "false" in compact_source().lower()
