#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/"tools"/"blender_retarget_rocketbox_walk.py"

TARGET_BONES = (
    "Bip01 Pelvis",
    "Bip01 Spine",
    "Bip01 Spine1",
    "Bip01 Spine2",
    "Bip01 Neck",
    "Bip01 Head",
    "Bip01 REye",
    "Bip01 LEye",
    "Bip01 MJaw",
    "Bip01 MBottomLip",
    "Bip01 MTongue",
    "Bip01 LMouthBottom",
    "Bip01 RMouthBottom",
    "Bip01 RMasseter",
    "Bip01 LMasseter",
    "Bip01 MUpperLip",
    "Bip01 RCaninus",
    "Bip01 LCaninus",
    "Bip01 REyeBlinkBottom",
    "Bip01 LEyeBlinkBottom",
    "Bip01 RUpperlip",
    "Bip01 LUpperlip",
    "Bip01 RMouthCorner",
    "Bip01 LMouthCorner",
    "Bip01 RCheek",
    "Bip01 LCheek",
    "Bip01 REyeBlinkTop",
    "Bip01 LEyeBlinkTop",
    "Bip01 RInnerEyebrow",
    "Bip01 LInnerEyebrow",
    "Bip01 MMiddleEyebrow",
    "Bip01 ROuterEyebrow",
    "Bip01 LOuterEyebrow",
    "Bip01 MNose",
    "Bip01 L Clavicle",
    "Bip01 L UpperArm",
    "Bip01 L Forearm",
    "Bip01 L Hand",
    "Bip01 L Finger0",
    "Bip01 L Finger01",
    "Bip01 L Finger02",
    "Bip01 L Finger1",
    "Bip01 L Finger11",
    "Bip01 L Finger12",
    "Bip01 L Finger2",
    "Bip01 L Finger21",
    "Bip01 L Finger22",
    "Bip01 L Finger3",
    "Bip01 L Finger31",
    "Bip01 L Finger32",
    "Bip01 L Finger4",
    "Bip01 L Finger41",
    "Bip01 L Finger42",
    "Bip01 R Clavicle",
    "Bip01 R UpperArm",
    "Bip01 R Forearm",
    "Bip01 R Hand",
    "Bip01 R Finger0",
    "Bip01 R Finger01",
    "Bip01 R Finger02",
    "Bip01 R Finger1",
    "Bip01 R Finger11",
    "Bip01 R Finger12",
    "Bip01 R Finger2",
    "Bip01 R Finger21",
    "Bip01 R Finger22",
    "Bip01 R Finger3",
    "Bip01 R Finger31",
    "Bip01 R Finger32",
    "Bip01 R Finger4",
    "Bip01 R Finger41",
    "Bip01 R Finger42",
    "Bip01 L Thigh",
    "Bip01 L Calf",
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Thigh",
    "Bip01 R Calf",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)

CORE_BONES = (
    "Bip01 Pelvis",
    "Bip01 Spine",
    "Bip01 Spine1",
    "Bip01 Spine2",
    "Bip01 Neck",
    "Bip01 Head",
    "Bip01 L Clavicle",
    "Bip01 L UpperArm",
    "Bip01 L Forearm",
    "Bip01 L Hand",
    "Bip01 R Clavicle",
    "Bip01 R UpperArm",
    "Bip01 R Forearm",
    "Bip01 R Hand",
    "Bip01 L Thigh",
    "Bip01 L Calf",
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Thigh",
    "Bip01 R Calf",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)

IMMUTABLE_HASH_KEYS = (
    "avatar_fbx",
    "motion_fbx",
    "source_review",
    "body_color_texture",
    "head_color_texture",
    "opacity_color_texture",
    "retarget_glb",
)


def renderer_source():
    assert SCRIPT.is_file()
    return SCRIPT.read_text(encoding="utf-8")


def tuple_constant(name):
    tree = ast.parse(renderer_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def function_node(name):
    tree = ast.parse(renderer_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def compact_source():
    return "".join(renderer_source().split())


def test_script_exposes_the_pinned_cli_and_source_review_gate():
    source = renderer_source()

    for option in (
        "--asset-id",
        "--avatar-fbx",
        "--texture-dir",
        "--texture-prefix",
        "--motion-fbx",
        "--source-review-json",
        "--output-dir",
    ):
        assert option in source
    assert "assert_source_review_approved(" in source
    assert 'review["asset_id"]' in source


def test_script_maps_all_exact_target_bones_and_rejects_bad_rigs():
    source = renderer_source()

    assert tuple_constant("TARGET_BONES") == TARGET_BONES
    assert tuple_constant("CORE_BONES") == CORE_BONES
    assert "len(TARGET_BONES) == 80" in source
    assert '"Nub"' in source
    assert "source_only_bones" in source
    assert "hierarchy_mismatches" in source


def test_script_uses_absolute_source_body_pose_instead_of_first_frame_rest_delta():
    source = compact_source()
    bake_source = "".join(
        ast.get_source_segment(renderer_source(), function_node("bake_target_action")).split()
    )

    assert "ABSOLUTE_POSE_BONES=CORE_BONES" in source
    assert "source_pb.matrix.to_quaternion().normalized()" in source
    assert "source_rest_local.inverted()@source_pose_local" not in source
    assert "source_pose_rotation" in source
    assert "target_rest_local.translation" in source
    assert "ifnameinABSOLUTE_POSE_BONES" in source
    assert "target_armature_matrix=Matrix.LocRotScale(" in source
    assert (
        "target_pb.matrix=target_armature_matrix"
        "bpy.context.view_layer.update()"
    ) in bake_source
    assert "parent_first_bones" in source
    assert "target_pb.matrix=target_armature_matrix" in source


def test_script_reports_shortest_rest_angles_and_reconstructed_target_facing():
    source = compact_source()

    assert "defshortest_rotation_angle(" in source
    assert "min(angle,2.0*math.pi-angle)" in source
    assert "target_root_quaternion@Vector((1.0,0.0,0.0))" in source


def test_facing_gate_reconstructs_source_motion_and_preserves_forward_floor():
    source = compact_source()
    reconstruction_tolerance = tuple_constant("FACING_RECONSTRUCTION_TOLERANCE")
    forward_floor = tuple_constant("FACING_FORWARD_DOT_FLOOR")
    measured_source_dot = 0.964627385
    measured_target_dot = 0.964627802
    double_axis_target_dot = 0.0
    reversed_target_dot = -measured_target_dot

    assert reconstruction_tolerance <= 1.0e-5
    assert forward_floor >= 0.90
    assert abs(measured_target_dot - measured_source_dot) <= reconstruction_tolerance
    assert measured_target_dot >= forward_floor
    assert abs(double_axis_target_dot - measured_source_dot) > reconstruction_tolerance
    assert double_axis_target_dot < forward_floor
    assert abs(reversed_target_dot - measured_source_dot) > reconstruction_tolerance
    assert reversed_target_dot < forward_floor
    assert "source_facing_dots=[]" in source
    assert "target_facing_dots=[]" in source
    assert "source_facing_dots.append(float(source_facing.dot(source_travel_unit)))" in source
    assert "target_facing_dots.append(float(target_facing.dot(travel_unit)))" in source
    assert "maximum_facing_reconstruction_error=max(" in source
    assert "ifmaximum_facing_reconstruction_error>FACING_RECONSTRUCTION_TOLERANCE" in source
    assert "ifmin(target_facing_dots)<FACING_FORWARD_DOT_FLOOR" in source
    assert '"source_minimum_facing_travel_dot"' in source
    assert '"source_maximum_facing_travel_dot"' in source
    assert '"target_minimum_facing_travel_dot"' in source
    assert '"target_maximum_facing_travel_dot"' in source
    assert '"maximum_facing_reconstruction_error"' in source


def test_script_rebases_object_root_once_without_axis_or_scale_changes():
    source = compact_source()

    assert "AXIS_MAP=Matrix.Identity(3)" in source
    assert "ROOT_SCALE=1.0" in source
    assert "source_location-source_frame_one_location" in source
    assert "helper_basis.inverted()@source_quaternion" in source
    assert "target_base_quaternion@root_motion_quaternion" in source
    assert "target_armature.scale=target_base_scale" in source
    assert "math.radians(180" not in source
    assert "target_base_quaternion@root_displacement" not in source


def test_script_preserves_target_mesh_materials_weights_and_facial_bones():
    source = renderer_source()

    assert "import_avatar(" in source
    assert "reconnect_official_materials(" in source
    assert ".materials.clear(" not in source
    for metric in (
        "vertex_count",
        "polygon_count",
        "uv_layer_count",
        "material_slot_count",
        "vertex_group_count",
        "bone_count",
        "material_slot_names",
    ):
        assert metric in source
    assert "unmapped_target_bones" in source
    assert "Matrix.Identity(4)" in source
    assert "def validate_official_material_bindings" in source
    assert "material_uses_color_as_alpha(" in source
    assert "official_color_image_names" in source


def test_script_authenticates_every_consumed_official_texture_before_import():
    source = renderer_source()

    assert tuple_constant("CONSUMED_TEXTURE_SUFFIXES") == (
        "body_color",
        "body_normal",
        "body_specular",
        "head_color",
        "head_normal",
        "head_specular",
        "opacity_color",
    )
    assert "approved texture hash mismatch" in source
    assert "for suffix in CONSUMED_TEXTURE_SUFFIXES" in source


def test_script_invalidates_readiness_before_blender_outputs_and_can_fail_after_export():
    source = renderer_source()
    compact = compact_source()
    main_source = ast.get_source_segment(source, function_node("main"))

    assert 'READINESS_FILES=("retarget_manifest.json","motion_review.json")' in compact
    assert "def invalidate_review_readiness(" in source
    assert "os.unlink(" in source
    assert "ROCKETBOX_RETARGET_FAIL_AFTER_EXPORT" in source
    assert main_source.index("invalidate_review_readiness(args.output_dir)") < main_source.index(
        "bpy.ops.wm.read_factory_settings(use_empty=True)"
    )


def test_script_uses_unique_same_directory_json_replacement():
    source = renderer_source()

    assert "tempfile.NamedTemporaryFile(" in source
    assert "dir=path.parent" in source
    assert "os.replace(" in source
    assert 'with_suffix(path.suffix + ".tmp")' not in source
    for message in ("Writing JSON", "Saving target blend", "Exporting target GLB"):
        assert message in source


def test_script_validates_roundtrip_vertex_groups_and_normalized_weights():
    source = renderer_source()

    assert "KDTree" in source
    assert "SKIN_POSITION_TOLERANCE_M" in source
    assert "SKIN_WEIGHT_L1_TOLERANCE" in source
    assert "tuple(group.name for group in mesh.vertex_groups) != TARGET_BONES" in source
    assert "vertices_without_influences" in source
    assert "maximum_weight_sum_error" in source
    assert "maximum_weight_l1_error" in source
    assert "mapped_original_vertex_count" in source
    assert "unmapped_original_vertex_count" in source
    assert "seam mapping did not cover every original vertex" not in source
    assert '"skin_weight_validation"' in source


def test_script_validates_semantic_gltf_material_bindings():
    source = renderer_source()

    for field in (
        "baseColorTexture",
        "normalTexture",
        "KHR_materials_specular",
        "specularTexture",
        "alphaMode",
        "BLEND",
    ):
        assert field in source
    assert "material_binding_errors" in source
    assert '"semantic_material_bindings"' in source


def test_script_creates_and_keyframes_a_new_target_action():
    source = renderer_source()

    assert "bpy.data.actions.new(" in source
    assert "target_armature.animation_data_create()" in source
    assert "target_armature.animation_data.action = target_action" in source
    assert 'keyframe_insert(data_path="location"' in source
    assert 'keyframe_insert(data_path="rotation_quaternion"' in source
    assert 'keyframe_insert(data_path="scale"' in source
    assert "interpolation = \"LINEAR\"" in source


def test_script_saves_target_only_blend_exports_glb_and_reimports_at_30_fps():
    source = renderer_source()

    assert "save_as_mainfile" in source
    assert "save_version = 0" in source
    assert "select_target_only" in source
    assert "export_scene.gltf" in source
    for option in (
        'export_format="GLB"',
        "use_selection=True",
        "export_animations=True",
        'export_animation_mode="ACTIVE_ACTIONS"',
        "export_force_sampling=True",
        "export_skins=True",
        "export_texcoords=True",
        "export_normals=True",
    ):
        assert option in source
    assert "read_factory_settings(use_empty=True)" in source
    assert "import_scene.gltf" in source
    assert "fps = 30" in source
    assert "roundtrip" in source


def test_script_writes_metrics_and_the_tightened_stage_manifest():
    source = renderer_source()
    build_manifest = function_node("build_manifest")

    hash_assignment = next(
        node
        for node in ast.walk(build_manifest)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "immutable_input_hashes"
            for target in node.targets
        )
    )
    assert isinstance(hash_assignment.value, ast.Dict)
    assert tuple(ast.literal_eval(key) for key in hash_assignment.value.keys) == (
        IMMUTABLE_HASH_KEYS
    )
    assert all(
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "sha256_file"
        for value in hash_assignment.value.values
    )

    returned_manifest = next(
        node.value
        for node in ast.walk(build_manifest)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    fields = {
        ast.literal_eval(key): value
        for key, value in zip(returned_manifest.keys, returned_manifest.values)
    }
    assert ast.literal_eval(fields["schema_version"]) == "rocketbox_retarget_manifest_v1"
    assert ast.literal_eval(fields["stage"]) == "retargeted"
    binding = {
        ast.literal_eval(key): ast.unparse(value)
        for key, value in zip(fields["binding"].keys, fields["binding"].values)
    }
    assert binding == {
        "target_asset_id": "args.asset_id",
        "target_mesh_bound": "True",
        "official_textures_attached": "True",
    }

    assert tuple_constant("IMMUTABLE_HASH_KEYS") == IMMUTABLE_HASH_KEYS
    assert '"schema_version": "rocketbox_retarget_manifest_v1"' in source
    assert '"stage": "retargeted"' in source
    assert '"target_asset_id": args.asset_id' in source
    assert '"target_mesh_bound": True' in source
    assert '"official_textures_attached": True' in source
    assert '"retarget_metrics.json"' in source
    assert '"retarget_manifest.json"' in source
    assert "ROCKETBOX_RETARGET_OK asset_id=" in source
    assert "_SHA256_RE.fullmatch" in source


def test_script_computes_loop_residual_from_baked_target_root_motion():
    source = compact_source()

    assert "baked_root_locations" in source
    assert "actual_cycle_displacement" in source
    assert "actual_cycle_displacement-expected_cycle_displacement" in source


def test_new_python_files_follow_repository_style_contract():
    copyright_lines = (
        "Copyright (c) 2025 The SPEAR Development Team",
        "Copyright (c) 2022 Intel",
    )
    forbidden_strings = (
        "# " + "noqa",
        "from dataclasses import " + "dataclass",
        "@data" + "class",
        "ArgumentParser(" + "description=",
    )
    for path in (SCRIPT, Path(__file__)):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for line in copyright_lines:
            assert line in source
        for forbidden in forbidden_strings:
            assert forbidden not in source
        assert not any(isinstance(node, ast.AnnAssign) for node in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is None
                args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                assert all(arg.annotation is None for arg in args)
