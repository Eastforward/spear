#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

import ast
import hashlib
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from tools.human_part_transfer import (
    HumanRegion,
    transfer_human_weights,
)


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/"tools"/"blender_bind_hy3d_to_rocketbox.py"


def script_source():
    assert SCRIPT.is_file(), f"missing Task 3 Blender binder: {SCRIPT}"
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
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def pure_function(name):
    namespace = {
        "np": np,
        "TARGET_ROTATE_Z_DEG": 0.0,
        "GLTF_MIN_INFLUENCE": 0.0001,
    }
    exec(function_source(name), namespace)
    return namespace[name]


def ground_cleanup_functions():
    namespace = {
        "np": np,
        "human_ground_artifact_mask": __import__(
            "tools.human_part_transfer", fromlist=["human_ground_artifact_mask"]
        ).human_ground_artifact_mask,
    }
    for name in (
        "GROUND_MAX_CENTER_HEIGHT_RATIO",
        "GROUND_MAX_COMPONENT_HEIGHT_RATIO",
        "GROUND_MIN_HORIZONTAL_SPREAD_RATIO",
        "GROUND_MIN_VERTICES",
        "GROUND_COPLANAR_HEIGHT_RATIO",
        "RESIDUAL_ROBUST_QUANTILE",
        "RESIDUAL_ROBUST_MARGIN_RATIO",
    ):
        namespace[name] = module_constant(name)
    exec(function_source("connected_vertex_components"), namespace)
    exec(function_source("ground_artifact_cleanup_mask"), namespace)
    exec(function_source("validate_residual_human_components"), namespace)
    return namespace


def box_mesh():
    vertices = np.array(
        (
            (-0.5, -0.2, 0.0),
            (0.5, -0.2, 0.0),
            (0.5, 0.2, 0.0),
            (-0.5, 0.2, 0.0),
            (-0.5, -0.2, 2.0),
            (0.5, -0.2, 2.0),
            (0.5, 0.2, 2.0),
            (-0.5, 0.2, 2.0),
        ),
        dtype=np.float64,
    )
    faces = np.array(
        (
            (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
        ),
        dtype=np.int64,
    )
    return vertices, faces


def plane_grid(size, span, z):
    coordinates = np.linspace(-span*0.5, span*0.5, size)
    vertices = np.array(
        [(x, y, z) for y in coordinates for x in coordinates],
        dtype=np.float64,
    )
    faces = []
    for row in range(size - 1):
        for column in range(size - 1):
            first = row*size + column
            faces.extend(
                (
                    (first, first + 1, first + size + 1),
                    (first, first + size + 1, first + size),
                )
            )
    return vertices, np.asarray(faces, dtype=np.int64)


def cylinder_mesh(rings=20, segments=16):
    vertices = []
    for ring in range(rings):
        z = 2.0*ring/(rings - 1)
        for segment in range(segments):
            angle = 2.0*np.pi*segment/segments
            vertices.append((0.3*np.cos(angle), 0.2*np.sin(angle), z))
    faces = []
    for ring in range(rings - 1):
        for segment in range(segments):
            following = (segment + 1) % segments
            first = ring*segments + segment
            second = ring*segments + following
            third = (ring + 1)*segments + following
            fourth = (ring + 1)*segments + segment
            faces.extend(((first, second, third), (first, third, fourth)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def combine_meshes(meshes):
    vertices = []
    faces = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + offset)
        offset += len(mesh_vertices)
    return np.vstack(vertices), np.vstack(faces)


def filtered_mesh(vertices, faces, remove_mask):
    keep = ~np.asarray(remove_mask, dtype=bool)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    kept_faces = faces[np.all(keep[faces], axis=1)]
    return vertices[keep], remap[kept_faces]


def read_obj_mesh(path):
    vertices = []
    faces = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                face = tuple(int(item.split("/", 1)[0]) - 1 for item in line.split()[1:])
                assert len(face) == 3
                faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_script_exposes_exact_cli_and_artifact_contract():
    parse = function_source("parse_args")

    for option in (
        "--asset-id",
        "--baseline-dir",
        "--hy3d-dir",
        "--idle-motion-fbx",
        "--output-dir",
    ):
        assert option in parse
    for filename in (
        "cleaned.obj",
        "bound.blend",
        "bound_walk.glb",
        "bound_idle.glb",
        "bind_metrics.json",
        "bind_manifest.json",
    ):
        assert filename in script_source()


def test_script_uses_only_the_task2_human_transfer_contract():
    source = script_source()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "tools.human_part_transfer" in imported_modules
    for helper in (
        "human_ground_artifact_mask",
        "cross_limb_bridge_face_mask",
        "target_regions_from_capsules",
        "transfer_human_weights",
        "collapse_finger_weights_to_palms",
    ):
        assert helper in source
    for forbidden in ("mixamo", "proxy", "crop", "automatic_weights"):
        assert all(forbidden not in module.lower() for module in imported_modules)
        assert forbidden not in source.lower()


def test_script_authenticates_and_opens_the_immutable_baseline_before_rest_capture():
    source = script_source()
    run = function_source("consume_snapshot_binding")
    capture = function_source("capture_rocketbox_source")

    assert "baseline_manifest.json" in source
    assert "rocketbox_baseline_manifest_v1" in source
    assert "retarget.blend" in source
    assert "sha256_file(" in source
    assert "size" in function_source("validate_baseline_inputs")
    assert "bpy.ops.wm.open_mainfile(" in run
    assert 'armature.data.pose_position = "REST"' in capture
    assert "bpy.context.view_layer.update()" in capture
    assert "len(armature.data.bones) != 80" in capture
    assert run.index("open_mainfile(") < run.index("capture_rocketbox_source(")
    assert run.index("capture_rocketbox_source(") < run.index("import_hy3d_obj(")


def test_baseline_root_manifest_and_motion_are_exactly_pinned():
    source = script_source()
    validate = function_source("validate_baseline_inputs")

    assert 'Path("/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1")' in compact_source()
    assert module_constant("BASELINE_MANIFEST_SHA256") == (
        "b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922"
    )
    assert 'manifest.get("baseline_id") != "rocketbox_neutral_walk_v1"' in validate
    assert 'manifest.get("schema_version") != "rocketbox_baseline_manifest_v1"' in validate
    assert 'manifest.get("motion") != "walk_neutral"' in validate
    assert "args.baseline_dir" in validate
    assert "CANONICAL_BASELINE_ROOT" in validate


def test_rest_capture_publishes_world_floor_not_armature_local_floor():
    capture = function_source("capture_rocketbox_source")
    run = function_source("consume_snapshot_binding")

    assert "world_vertices = np.array(" in capture
    assert "source_mesh.matrix_world @ vertex.co" in capture
    assert '"floor_z_m": float(world_vertices[:, 2].min())' in capture
    assert 'floor_z_m = float(source["floor_z_m"])' in run
    assert 'source["bbox"][0, 2]' not in run


def test_cleanup_uses_both_proven_masks_and_records_component_metrics():
    source = script_source()
    cleanup = function_source("cleanup_target_geometry")
    consume = function_source("consume_snapshot_binding")

    assert "human_ground_artifact_mask(" in cleanup
    assert "cross_limb_bridge_face_mask(" in cleanup
    assert "component_metrics(" in cleanup
    assert '"before"' in cleanup
    assert '"after"' in cleanup
    assert '"removed_vertices"' in cleanup
    assert '"removed_faces"' in cleanup
    for forbidden in ("bisect", "decimate", "voxel", "boolean"):
        assert forbidden not in cleanup.lower()
    assert "delete_masked_vertices(" in cleanup
    assert "delete_masked_faces(" in cleanup
    assert "unmatched" in source
    assert "part-aware transfer left unmatched target vertices" in source
    assert consume.index("cleanup_import_ground_artifacts(") < consume.index(
        "move_target_to_armature_space("
    )
    assert consume.index("move_target_to_armature_space(") < consume.index(
        "cleanup_target_geometry("
    )
    assert consume.index("cleanup_target_geometry(") < consume.index(
        "bind_target_mesh("
    )
    assert "validate_residual_human_components(" in function_source(
        "cleanup_import_ground_artifacts"
    )


def test_ground_disc_is_removed_before_human_bbox_and_body_is_preserved():
    functions = ground_cleanup_functions()
    cleanup_mask = functions["ground_artifact_cleanup_mask"]
    validate_residual = functions["validate_residual_human_components"]
    body = box_mesh()
    disc = plane_grid(7, 6.0, -0.05)
    fragment = (
        np.array(
            ((2.1, -0.1, -0.05), (2.3, -0.1, -0.05),
             (2.3, 0.1, -0.05), (2.1, 0.1, -0.05)),
            dtype=np.float64,
        ),
        np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    vertices, faces = combine_meshes((body, disc, fragment))

    remove_mask, metrics = cleanup_mask(vertices, faces)

    assert not remove_mask[:len(body[0])].any()
    assert remove_mask[len(body[0]):].all()
    assert metrics["initial_ground_vertices"] == len(disc[0])
    assert metrics["coplanar_expanded_vertices"] == len(fragment[0])
    assert metrics["removed_vertices"] == len(disc[0]) + len(fragment[0])
    assert metrics["cleaned_bbox_extent"] == pytest.approx((1.0, 0.4, 2.0))
    kept_vertices, kept_faces = filtered_mesh(vertices, faces, remove_mask)
    residual = validate_residual(kept_vertices, kept_faces)
    assert residual["far_component_count"] == 0
    assert residual["large_flat_vertex_count"] == 0


def test_low_floor_card_narrower_than_body_span_is_still_removed():
    functions = ground_cleanup_functions()
    cleanup_mask = functions["ground_artifact_cleanup_mask"]
    body = box_mesh()
    floor_card = plane_grid(7, 0.55, -0.05)
    vertices, faces = combine_meshes((body, floor_card))

    remove_mask, metrics = cleanup_mask(vertices, faces)

    assert not remove_mask[:len(body[0])].any()
    assert remove_mask[len(body[0]):].all()
    assert metrics["cleaned_bbox_extent"] == pytest.approx((1.0, 0.4, 2.0))


def test_cleanup_hard_fails_for_residual_far_or_large_flat_components():
    functions = ground_cleanup_functions()
    validate_residual = functions["validate_residual_human_components"]
    body = cylinder_mesh()
    far = (
        np.array(
            ((1000.0, 1000.0, 1000.0), (1000.1, 1000.0, 1000.0),
             (1000.0, 1000.1, 1000.0)),
            dtype=np.float64,
        ),
        np.array(((0, 1, 2),), dtype=np.int64),
    )
    disc = plane_grid(7, 6.0, -0.05)

    with pytest.raises(RuntimeError, match="far component"):
        validate_residual(*combine_meshes((body, far)))
    with pytest.raises(RuntimeError, match="large flat component"):
        validate_residual(*combine_meshes((body, disc)))


@pytest.mark.parametrize(
    "asset_id, expected_raw_extents, expected_removed, expected_cleaned_extents",
    (
        (
            "rocketbox_male_adult_01",
            (1.995224, 0.570377, 1.995226),
            22065,
            (0.505765, 0.136827, 0.570377),
        ),
        (
            "rocketbox_female_adult_01",
            (1.276626, 1.992611, 0.451099),
            2552,
            (1.276626, 0.349112, 1.980970),
        ),
    ),
)
def test_current_real_hy3d_ground_cleanup_preserves_only_human_bbox(
    asset_id,
    expected_raw_extents,
    expected_removed,
    expected_cleaned_extents,
):
    candidate = REPO/"tmp"/"hy3d_rocketbox_spike_v1"/asset_id/"hy3d_textured.obj"
    if not candidate.is_file():
        pytest.skip("current authenticated Hunyuan candidate is not present")
    functions = ground_cleanup_functions()
    cleanup_mask = functions["ground_artifact_cleanup_mask"]
    validate_residual = functions["validate_residual_human_components"]
    raw_vertices, faces = read_obj_mesh(candidate)
    importer_matrix = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    canonical_vertices = raw_vertices @ importer_matrix.T

    remove_mask, metrics = cleanup_mask(canonical_vertices, faces)
    cleaned_vertices, cleaned_faces = filtered_mesh(
        canonical_vertices, faces, remove_mask
    )

    assert np.ptp(raw_vertices, axis=0) == pytest.approx(
        expected_raw_extents, abs=1.0e-6
    )
    assert int(remove_mask.sum()) == expected_removed
    assert metrics["cleaned_bbox_extent"] == pytest.approx(
        expected_cleaned_extents, abs=1.0e-6
    )
    residual = validate_residual(cleaned_vertices, cleaned_faces)
    assert residual["far_component_count"] == 0
    assert residual["large_flat_vertex_count"] == 0


def test_rest_source_excludes_only_zero_area_transfer_faces():
    filter_faces = pure_function("nondegenerate_transfer_faces")
    vertices = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    faces = np.array(((0, 1, 2), (0, 1, 3)))

    usable, metrics = filter_faces(vertices, faces)

    assert usable.tolist() == [[0, 1, 2]]
    assert metrics == {
        "original_face_count": 2,
        "usable_face_count": 1,
        "zero_area_face_count": 1,
    }
    assert "nondegenerate_transfer_faces(" in function_source(
        "capture_rocketbox_source"
    )


def test_alignment_is_zero_degree_single_scale_xy_center_and_floor_z():
    source = script_source()
    align = pure_function("uniform_bbox_alignment")
    target = np.array(((-1.0, -2.0, 3.0), (1.0, 2.0, 7.0)))
    source_bbox = np.array(((10.0, 20.0, -4.0), (14.0, 26.0, 4.0)))

    aligned, metrics = align(target, source_bbox)

    assert module_constant("TARGET_ROTATE_Z_DEG") == 0.0
    assert metrics["uniform_scale"] == 2.0
    assert metrics["floor_z_m"] == -4.0
    assert np.allclose(aligned[:, :2].mean(axis=0), (12.0, 23.0))
    assert np.isclose(aligned[:, 2].min(), -4.0)
    assert "math.radians(-90" not in compact_source()
    assert "scale = source_extent[2] / target_extent[2]" in function_source(
        "uniform_bbox_alignment"
    )


def test_reviewed_raw_axes_drive_fixed_blender_import_and_are_published():
    contracts = module_constant("RAW_HY3D_AXIS_CONTRACTS")
    expected = {
        asset_id: {
            "source_up_axis": "Y",
            "source_front_axis": "Z",
            "import_forward_axis": "NEGATIVE_Z",
            "import_up_axis": "Y",
            "expected_basis_matrix": (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0),
            ),
        }
        for asset_id in (
            "rocketbox_male_adult_01",
            "rocketbox_female_adult_01",
        )
    }
    importer = function_source("import_hy3d_obj")
    manifest = function_source("build_bind_manifest")

    assert contracts == expected
    assert 'forward_axis=contract["import_forward_axis"]' in importer
    assert 'up_axis=contract["import_up_axis"]' in importer
    assert '"raw_extents": raw_extents.tolist()' in importer
    assert '"target_rotate_z_deg": TARGET_ROTATE_Z_DEG' in importer
    assert "argmax" not in importer
    assert "detected_up" not in importer
    assert 'manifest["axis_contract"] = axis_contract' in manifest
    assert '"axis_contract": axis_contract' in function_source(
        "consume_snapshot_binding"
    )


def test_reviewed_import_basis_maps_front_and_up_and_rejects_old_matrix():
    namespace = {"np": np}
    exec(function_source("validate_import_basis_matrix"), namespace)
    validate = namespace["validate_import_basis_matrix"]
    contract = module_constant("RAW_HY3D_AXIS_CONTRACTS")[
        "rocketbox_male_adult_01"
    ]
    expected = np.asarray(contract["expected_basis_matrix"], dtype=np.float64)

    metrics = validate(expected, contract)

    assert np.allclose(expected @ (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    assert np.allclose(expected @ (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert np.isclose(np.linalg.det(expected), 1.0)
    assert metrics["canonical_front_vector"] == pytest.approx((0.0, -1.0, 0.0))
    assert metrics["canonical_up_vector"] == pytest.approx((0.0, 0.0, 1.0))
    assert metrics["basis_determinant"] == pytest.approx(1.0)
    old_wrong_matrix = np.array(
        ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    with pytest.raises(RuntimeError, match="reviewed basis"):
        validate(old_wrong_matrix, contract)


def test_import_cleanup_and_binding_preserve_uvs_and_build_hunyuan_pbr():
    source = script_source()
    material = function_source("assign_hunyuan_pbr_material")

    assert "hy3d_textured.obj" in source
    assert "hy3d_diffuse.jpg" in source
    assert "hy3d_metallic.jpg" in source
    assert "hy3d_roughness.jpg" in source
    assert "uv_layers" in source
    assert "material_index" in source
    assert 'nodes.new("ShaderNodeTexImage")' in material
    assert 'nodes.new("ShaderNodeBsdfPrincipled")' in material
    assert 'principled.inputs["Base Color"]' in material
    assert 'principled.inputs["Metallic"]' in material
    assert 'principled.inputs["Roughness"]' in material
    assert 'colorspace_settings.name = "Non-Color"' in source
    assert "target.data.materials.clear()" in material
    assert "target.data.materials.append(material)" in material
    assert "material_slot_remove_unused" not in material
    assert "len(target.material_slots) != 1" in material
    assert "bpy.ops.wm.obj_export(" in source


def test_bound_blend_packs_and_revalidates_the_complete_pbr_graph():
    pack = function_source("pack_pbr_images")
    validate = function_source("validate_packed_pbr_material")
    consume = function_source("consume_snapshot_binding")

    assert module_constant("PBR_NODE_CONTRACT") == {
        "diffuse": {
            "node_name": "Hunyuan Diffuse",
            "image_name": "hy3d_diffuse",
            "principled_input": "Base Color",
            "colorspace": "sRGB",
        },
        "metallic": {
            "node_name": "Hunyuan Metallic",
            "image_name": "hy3d_metallic",
            "principled_input": "Metallic",
            "colorspace": "Non-Color",
        },
        "roughness": {
            "node_name": "Hunyuan Roughness",
            "image_name": "hy3d_roughness",
            "principled_input": "Roughness",
            "colorspace": "Non-Color",
        },
    }
    assert "image.pack()" in pack
    assert "image.packed_file is None" in pack
    assert "image.packed_file.size <= 0" in pack
    for token in (
        "image.packed_file is None",
        "image.packed_file.size <= 0",
        "image.size[0] <= 0",
        "image.size[1] <= 0",
        "image.has_data",
        "image.pixels[0]",
        "colorspace_settings.name",
        "material.node_tree.links",
        "link.from_node",
        "link.to_node",
        "link.to_socket",
    ):
        assert token in validate
    assert consume.index("save_bound_blend(") < consume.index("load_saved_target(")
    assert consume.index("load_saved_target(") < consume.index(
        "validate_packed_pbr_material("
    )
    assert '"bound_blend_pbr": bound_blend_pbr' in consume


def test_cleaned_obj_is_explicitly_geometry_only_without_mtl_sidecars():
    export = function_source("export_cleaned_obj")
    manifest = function_source("build_bind_manifest")
    consume = function_source("consume_snapshot_binding")

    assert "export_uv=True" in export
    assert "export_normals=True" in export
    assert "export_materials=False" in export
    assert "export_pbr_extensions=False" in export
    assert 'path.with_suffix(".mtl")' in export
    assert 'line.startswith(("mtllib ", "usemtl "))' in export
    assert '"role": "geometry_only"' in export
    assert '"cleaned_obj_contract"' in manifest
    assert '"role": "geometry_only"' in manifest
    assert '"cleaned_obj": cleaned_obj_metrics' in consume


def test_binding_creates_exact_rocketbox_groups_one_modifier_and_palm_weights():
    source = script_source()
    bind = function_source("bind_target_mesh")
    validate = function_source("validate_bound_weights")

    assert "TARGET_BONES" in bind
    assert "transfer_with_distance_contract(" in bind
    assert "collapse_finger_weights_to_palms(" in bind
    assert "prune_gltf_influences(" in bind
    assert "target.vertex_groups.clear()" in bind
    assert 'target.modifiers.new(name="Rocketbox Armature", type="ARMATURE")' in bind
    assert "influence_count > 4" in validate
    assert "abs(weight_sum - 1.0)" in validate
    assert "zero-weight target vertex" in validate
    assert "finger vertex groups must be empty after palm collapse" in validate


def test_scale_derived_distance_rejects_a_far_detached_component():
    namespace = {
        "np": np,
        "transfer_human_weights": transfer_human_weights,
        "MAX_DISTANCE_HEIGHT_RATIO": 0.20,
    }
    exec(function_source("transfer_with_distance_contract"), namespace)
    transfer = namespace["transfer_with_distance_contract"]
    source = {
        "vertices": np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "faces": np.array(((0, 1, 2),)),
        "weights": np.ones((3, 1)),
        "group_names": ["Bip01 Pelvis"],
    }
    vertices = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1000.0, 1000.0, 1000.0),
            (1001.0, 1000.0, 1000.0),
            (1000.0, 1000.0, 1001.0),
        )
    )
    faces = np.array(((0, 1, 2), (3, 4, 5)))
    target = {
        "vertices": vertices,
        "faces": faces,
        "regions": np.full(6, int(HumanRegion.TORSO)),
    }
    _, incomplete = transfer_human_weights(
        source,
        target,
        max_distance=0.20,
        require_complete=False,
    )

    assert module_constant("MAX_DISTANCE_HEIGHT_RATIO") == 0.20
    assert incomplete["initial_matched"] == 3
    assert incomplete["initial_unmatched"] == 3
    assert incomplete["graph_filled"] == 0
    assert incomplete["unmatched"] == 3
    with pytest.raises(ValueError, match="incomplete human weight transfer"):
        transfer(source, target)


def test_distance_contract_is_passed_and_recorded_by_binding():
    bind = function_source("bind_target_mesh")

    assert "transfer_with_distance_contract(" in bind
    assert "max_distance=max_distance" in function_source(
        "transfer_with_distance_contract"
    )
    assert "require_complete=True" in function_source(
        "transfer_with_distance_contract"
    )
    assert '"source_height"' in function_source("transfer_with_distance_contract")
    assert '"max_distance"' in function_source("transfer_with_distance_contract")


def test_binding_prunes_the_blender_gltf_minimum_influence_before_groups():
    prune = pure_function("prune_gltf_influences")
    weights = np.array(((0.9999273964644139, 0.0000726035355861), (0.6, 0.4)))

    actual = prune(weights)

    assert module_constant("GLTF_MIN_INFLUENCE") == 0.0001
    assert np.allclose(actual, ((1.0, 0.0), (0.6, 0.4)))
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_walk_is_reused_and_idle_uses_the_proven_source_absolute_bake_helpers():
    source = script_source()
    idle = function_source("bake_idle_action")

    assert "from tools import blender_retarget_rocketbox_walk as retarget" in source
    for call in (
        "retarget.import_source_motion(",
        "retarget.cache_source_frames(",
        "retarget.validate_mapping(",
        "retarget.parent_first_names(",
        "retarget.bake_target_action(",
        "retarget.validate_action_ownership(",
        "retarget.remove_source_import(",
    ):
        assert call in idle
    assert 'f"{asset_id}_idle_neutral_01_retarget"' in idle
    assert "source.action" not in idle
    assert "walk_action = armature.animation_data.action" in source
    assert "approved baseline walk action" in source
    assert "capture_target_base_transform(" in idle
    assert "restore_target_base_transform(" in idle
    assert idle.index("capture_target_base_transform(") < idle.index(
        "retarget.cache_source_frames("
    )
    assert idle.index("retarget.cache_source_frames(") < idle.index(
        "restore_target_base_transform("
    )
    assert idle.index("restore_target_base_transform(") < idle.index(
        "retarget.bake_target_action("
    )


def test_idle_files_are_exact_official_byte_and_git_blob_pins(tmp_path):
    pins = module_constant("IDLE_PINS")
    assert pins == {
        "rocketbox_male_adult_01": {
            "filename": "m_idle_neutral_01.max.fbx",
            "size_bytes": 2418544,
            "sha256": "818cc185af21390575f7fbfdeb3012ba2ce5969fbcb220ea725a2617b339a6e2",
            "git_blob_sha1": "a2d92c3326a9c503af677c9fa6082387f060d6c4",
        },
        "rocketbox_female_adult_01": {
            "filename": "f_idle_neutral_01.max.fbx",
            "size_bytes": 2959360,
            "sha256": "fd68b33ea9e290dc734ca8c3a71ef5842bb2dfe719853ff84f6336d06d39fdcb",
            "git_blob_sha1": "aecf1d0089ccfc0c381d5395294bb1c8fe0e63ae",
        },
    }
    namespace = {"hashlib": hashlib, "Path": Path}
    exec(function_source("git_blob_sha1_file"), namespace)
    payload = tmp_path/"idle.fbx"
    payload.write_bytes(b"hello")
    expected = hashlib.sha1(b"blob 5\0hello").hexdigest()
    assert namespace["git_blob_sha1_file"](payload) == expected


def test_idle_validation_rejects_renamed_bytes_before_hash_acceptance():
    validate = function_source("validate_idle_motion")

    assert "CANONICAL_IDLE_ROOT/pin[\"filename\"]" in validate
    assert "path != expected_path" in validate
    assert 'pin["size_bytes"]' in validate
    assert 'pin["sha256"]' in validate
    assert 'pin["git_blob_sha1"]' in validate
    assert "git_blob_sha1_file(" in validate


def test_hy3d_validation_reuses_task1_gate_and_current_provenance():
    source = script_source()
    validate = function_source("validate_hy3d_inputs")

    assert "import hy3d_human_candidate as hy3d_contract" in source
    assert 'SPEAR_ROOT/"tmp"/"hy3d_rocketbox_spike_v1"' in source
    assert 'SPEAR_ROOT/"tmp"/"human_reference_review"' in source
    for call in (
        "contract.assert_generation_ready(",
        "contract.current_hunyuan_runtime_provenance(",
        "contract.verify_canonical_weights(",
        "contract.current_weight_manifest_sha256(",
    ):
        assert call in validate
    for field in (
        "candidate_sha256",
        "candidate_manifest_sha256",
        "source_sha256",
        "source_approval_sha256",
        "reference_review_sha256",
        "hunyuan_runtime_git_head",
        "hunyuan_runtime_fingerprint",
        "hunyuan_runtime_file_count",
        "weight_manifest_sha256",
        "seed",
        "steps",
        "guidance_scale",
        "usage_scope",
    ):
        assert field in source


def test_fabricated_self_consistent_hy3d_manifest_is_rejected():
    output_names = {
        "reference": "reference.png",
        "reference_rembg": "reference_rembg.png",
        "shape": "shape.glb",
        "paint_obj": "hy3d_textured.obj",
        "diffuse": "hy3d_diffuse.jpg",
        "metallic": "hy3d_metallic.jpg",
        "roughness": "hy3d_roughness.jpg",
    }
    namespace = {
        "Path": Path,
        "ASSET_SEEDS": {
            "rocketbox_male_adult_01": 4101,
            "rocketbox_female_adult_01": 7301,
        },
        "HY3D_FILENAMES": output_names,
        "HY3D_SCHEMA_VERSION": "hy3d_human_candidate_v1",
        "HY3D_USAGE_SCOPE": "technical_spike_only",
        "CANONICAL_MODEL_ROOT": Path("/data/models/hunyuan3d-2.1/hunyuan3d-2.1"),
        "WEIGHT_ROOT_HASH_MANIFEST": Path("/data/models/hunyuan3d-2.1/weights.sha256"),
    }
    exec(function_source("validate_hy3d_provenance_payload"), namespace)
    validate = namespace["validate_hy3d_provenance_payload"]
    approved = "a"*64
    fabricated = "b"*64
    runtime = {"git_head": "c"*40, "fingerprint": "d"*64, "file_count": 17}
    job = {
        "candidate_sha256": approved,
        "candidate_manifest_sha256": "e"*64,
        "source_sha256": "f"*64,
        "source_approval_sha256": "1"*64,
        "reference_review_sha256": "2"*64,
        "seed": 4101,
        "steps": 50,
        "guidance_scale": 5.0,
        "model_root": Path("/data/models/hunyuan3d-2.1/hunyuan3d-2.1"),
        "weight_root_hash_manifest": Path("/data/models/hunyuan3d-2.1/weights.sha256"),
        "hunyuan_runtime_git_head": runtime["git_head"],
        "hunyuan_runtime_fingerprint": runtime["fingerprint"],
        "hunyuan_runtime_file_count": runtime["file_count"],
    }
    manifest = {
        "schema_version": "hy3d_human_candidate_v1",
        "asset_id": "rocketbox_male_adult_01",
        "candidate_sha256": fabricated,
        "candidate_manifest_sha256": job["candidate_manifest_sha256"],
        "source_sha256": job["source_sha256"],
        "source_approval_sha256": job["source_approval_sha256"],
        "reference_review_sha256": job["reference_review_sha256"],
        "hunyuan_code_revision": runtime["git_head"],
        "hunyuan_runtime_git_head": runtime["git_head"],
        "hunyuan_runtime_fingerprint": runtime["fingerprint"],
        "hunyuan_runtime_file_count": runtime["file_count"],
        "weight_root": str(job["model_root"]),
        "weight_root_hash_manifest": str(job["weight_root_hash_manifest"]),
        "weight_manifest_sha256": "3"*64,
        "seed": 4101,
        "steps": 50,
        "guidance_scale": 5.0,
        "usage_scope": "technical_spike_only",
        "outputs": {
            role: {
                "path": filename,
                "sha256": fabricated if role == "reference" else "4"*64,
                "size_bytes": 10,
            }
            for role, filename in output_names.items()
        },
    }

    with pytest.raises(ValueError, match="approved candidate|candidate_sha256"):
        validate(
            manifest,
            "rocketbox_male_adult_01",
            job,
            runtime,
            "3"*64,
            "3"*64,
        )


def test_bound_blend_keeps_two_actions_and_each_glb_exports_one():
    source = script_source()
    validate_actions = function_source("validate_two_actions")
    export = function_source("export_single_action_glb")

    assert "len(bpy.data.actions) != 2" in validate_actions
    assert "walk_action == idle_action" in validate_actions
    assert "walk_action.use_fake_user = True" in source
    assert "idle_action.use_fake_user = True" in source
    assert "bpy.ops.wm.save_as_mainfile(" in source
    assert "def isolate_action_for_export" in source
    assert "bpy.data.actions.remove(" in function_source("isolate_action_for_export")
    assert "set(bpy.data.actions) != {action}" in export
    assert 'export_animation_mode="ACTIVE_ACTIONS"' in export
    assert "armature.animation_data.action = action" in export
    assert "bound_walk.glb" in source
    assert "bound_idle.glb" in source
    assert 'atomic_copy(walk_path, output_dir/"walk.glb")' not in source
    assert 'atomic_copy(idle_path, output_dir/"idle.glb")' not in source


def test_glbs_receive_structure_uv_pbr_skin_and_joint_roundtrip_checks():
    source = script_source()
    inspect = function_source("inspect_bound_glb")

    for token in (
        'payload.get("skins"',
        'payload.get("animations"',
        'primitive.get("attributes"',
        '"TEXCOORD_0"',
        '"JOINTS_0"',
        '"WEIGHTS_0"',
        '"baseColorTexture"',
        '"metallicRoughnessTexture"',
    ):
        assert token in inspect
    assert "retarget.capture_skin_contract(" in source
    assert "retarget.roundtrip_validate(" in source
    assert '"skin_weight_validation"' in source
    assert '"maximum_world_joint_error_m"' in source


def test_bind_manifest_matches_review_contract_and_hashes_current_sources():
    source = script_source()
    manifest = function_source("build_bind_manifest")
    descriptor = function_source("file_descriptor")

    assert '"schema_version": "hy3d_rocketbox_bind_v1"' in manifest
    assert '"asset_id": args.asset_id' in manifest
    assert '"reference"' in manifest
    assert '"glbs"' in manifest
    assert '"walk":file_descriptor(output_dir/"bound_walk.glb")' in compact_source()
    assert '"idle":file_descriptor(output_dir/"bound_idle.glb")' in compact_source()
    assert 'return {"filename": path.name, "sha256": sha256_file(path)}' in descriptor
    assert '"action_names"' in manifest
    assert '"walk": action_metrics["walk"]["action_name"]' in manifest
    assert '"idle": action_metrics["idle"]["action_name"]' in manifest
    assert '"floor_z_m": floor_z_m' in manifest
    assert 'source["floor_z_m"]' in source
    assert "baseline_manifest_sha256" in source
    assert "baseline_manifest_current_sha256" in source
    assert "hy3d_manifest_sha256" in source
    assert "hy3d_manifest_current_sha256" in source
    assert "idle_motion_fbx_sha256" in source
    assert "idle_motion_fbx_current_sha256" in source
    assert "source inputs changed during binding" in source


def test_private_snapshot_copy_is_nofollow_stable_and_aba_safe(tmp_path):
    namespace = {
        "hashlib": hashlib,
        "os": os,
        "Path": Path,
        "stat": stat,
    }
    exec(function_source("copy_authenticated_file"), namespace)
    copy_file = namespace["copy_authenticated_file"]
    source = tmp_path/"source.bin"
    source.write_bytes(b"authenticated-A")
    expected_sha256 = hashlib.sha256(b"authenticated-A").hexdigest()
    destination = tmp_path/"private"/"snapshot.bin"
    destination.parent.mkdir(mode=0o700)

    def replace_and_restore(path):
        held = path.with_name("held-A")
        os.replace(path, held)
        path.write_bytes(b"attacker-B")
        path.unlink()
        os.replace(held, path)

    record = copy_file(
        source,
        destination,
        expected_sha256,
        len(b"authenticated-A"),
        replace_and_restore,
    )

    assert destination.read_bytes() == b"authenticated-A"
    assert record["sha256"] == expected_sha256
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    source.write_bytes(b"attacker-B")
    rejected = tmp_path/"private"/"rejected.bin"
    with pytest.raises(RuntimeError, match="authenticated|hash|size"):
        copy_file(source, rejected, expected_sha256, len(b"authenticated-A"))
    assert not rejected.exists()


def test_run_consumes_only_private_snapshot_paths_and_always_cleans_it():
    source = script_source()
    run = function_source("run_binding")
    consume = function_source("consume_snapshot_binding")
    stage = function_source("stage_input_snapshot")
    copy_file = function_source("copy_authenticated_file")

    assert "tempfile.mkdtemp(" in stage
    assert "output_dir.parent" in stage
    assert "os.chmod(snapshot_root, 0o700)" in stage
    assert "copy_authenticated_file(" in stage
    assert "os.O_NOFOLLOW" in copy_file
    assert "os.O_DIRECTORY" in copy_file
    assert "os.fstat(" in copy_file
    assert "stage_input_snapshot(" in run
    assert "consume_snapshot_binding(" in run
    assert 'snapshot["paths"]["baseline_blend"]' in consume
    assert 'snapshot["paths"]["hy3d_paint_obj"]' in consume
    assert 'snapshot["paths"]["idle_motion_fbx"]' in consume
    assert 'snapshot["paths"]["hy3d_reference"]' in consume
    assert "finally:" in run
    assert "cleanup_input_snapshot(snapshot)" in run
    assert 'baseline["baseline_blend_path"]' not in consume


def test_snapshot_provenance_is_published_by_the_real_manifest_builder():
    manifest_source = function_source("build_bind_manifest")

    assert '"consumed_inputs": consumed_inputs' in manifest_source


def test_real_build_bind_manifest_has_the_exact_task5_consumer_shape():
    sha256 = "a"*64

    def descriptor(path):
        return {"filename": Path(path).name, "sha256": sha256}

    namespace = {"file_descriptor": descriptor}
    exec(function_source("build_bind_manifest"), namespace)
    build_manifest = namespace["build_bind_manifest"]
    action_metrics = {
        "walk": {
            "action_name": "approved_walk",
            "frame_start": 1,
            "frame_end": 33,
        },
        "idle": {
            "action_name": "baked_idle",
            "frame_start": 1,
            "frame_end": 351,
        },
    }
    axis_contract = {
        "source_up_axis": "Y",
        "source_front_axis": "Z",
        "import_forward_axis": "NEGATIVE_Z",
        "import_up_axis": "Y",
        "expected_basis_matrix": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
        "raw_extents": [1.995224, 0.570377, 1.995226],
        "target_rotate_z_deg": 0.0,
    }

    manifest = build_manifest(
        SimpleNamespace(asset_id="rocketbox_male_adult_01"),
        Path("/tmp/task3-bind"),
        action_metrics,
        {"source": "captured"},
        {"source_current": "verified"},
        -0.004898,
        {
            "baseline_blend": {
                "filename": "retarget.blend",
                "sha256": sha256,
                "size_bytes": 42,
            }
        },
        axis_contract,
    )

    assert manifest["glbs"] == {
        "walk": {"filename": "bound_walk.glb", "sha256": sha256},
        "idle": {"filename": "bound_idle.glb", "sha256": sha256},
    }
    assert all(set(value) == {"filename", "sha256"} for value in manifest["glbs"].values())
    assert manifest["action_names"] == {
        "walk": "approved_walk",
        "idle": "baked_idle",
    }
    assert manifest["action_names"]["walk"] != manifest["action_names"]["idle"]
    assert manifest["bound_blend"] == {
        "filename": "bound.blend",
        "sha256": sha256,
    }
    assert manifest["axis_contract"] == axis_contract
    assert manifest["cleaned_obj_contract"] == {
        "role": "geometry_only",
        "materials": False,
        "uv": True,
        "normals": True,
    }
    assert set(manifest["bound_blend"]) == {"filename", "sha256"}
    assert manifest["consumed_inputs"]["baseline_blend"]["sha256"] == sha256
    assert "actions" not in manifest

    legacy_manifest = build_manifest(
        SimpleNamespace(asset_id="rocketbox_male_adult_01"),
        Path("/tmp/task3-bind"),
        action_metrics,
        {"source": "captured"},
        {"source_current": "verified"},
        -0.004898,
        {"baseline_blend": {"filename": "retarget.blend", "sha256": sha256}},
    )
    assert "axis_contract" not in legacy_manifest


def test_readiness_is_invalidated_before_blender_and_manifest_is_atomic_last():
    source = script_source()
    main = function_source("main")
    run = function_source("consume_snapshot_binding")
    write = function_source("atomic_write_json")

    assert module_constant("READINESS_FILES") == (
        "bind_manifest.json",
        "review_manifest.json",
        "hy3d_rocketbox_review.json",
    )
    assert main.index("invalidate_readiness(args.output_dir)") < main.index("run_binding(args)")
    assert "except BaseException:" in main
    assert main.count("invalidate_readiness(args.output_dir)") == 2
    assert "tempfile.NamedTemporaryFile(" in write
    assert "dir=path.parent" in write
    assert "os.replace(" in write
    assert run.index("atomic_write_json(output_dir/\"bind_metrics.json\"") < run.index(
        "atomic_write_json(output_dir/\"bind_manifest.json\""
    )
    assert "HY3D_ROCKETBOX_BIND_FAIL_AFTER_EXPORT" in source


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
        "he" + "lp=",
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
