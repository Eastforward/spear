#!/usr/bin/env python3

"""Bind an authenticated I23D mesh directly to an approved Rocketbox rig."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
for import_dir in (SPEAR_ROOT, TOOLS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from tools import blender_bind_hy3d_to_rocketbox as direct
from tools import blender_fit_hy3d_to_rocketbox_template as template_fit
from tools import i23d_rocketbox_contract


BINDING_MODE = "direct_i23d_mesh_to_rocketbox_v1"
DIRECT_BIND_FACE_BUDGET = 120000
OUTPUT_FILENAMES = (
    "cleaned.obj",
    "bound.blend",
    "bound_walk.glb",
    "bound_idle.glb",
    "bind_metrics.json",
    "bind_manifest.json",
)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", choices=direct.EXPECTED_ASSET_IDS, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guide-glb", type=Path, required=True)
    parser.add_argument("--guide-manifest", type=Path, required=True)
    parser.add_argument(
        "--guide-backend",
        choices=tuple(i23d_rocketbox_contract.BACKEND_CONTRACTS),
        required=True,
    )
    parser.add_argument(
        "--front-axis", choices=("negative-y", "positive-y"), required=True
    )
    parser.add_argument("--reference-rgba", type=Path, required=True)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def validate_i23d_inputs(args):
    return template_fit.validate_i23d_inputs(args)


def material_image_nodes(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return []
    return [
        node
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    ]


def select_primary_textured_mesh(meshes):
    candidates = []
    for mesh in meshes:
        if len(mesh.data.uv_layers) < 1 or len(mesh.material_slots) != 1:
            continue
        material = mesh.material_slots[0].material
        if not material_image_nodes(material):
            continue
        candidates.append(mesh)
    if not candidates:
        raise RuntimeError("I23D GLB has no textured mesh with UVs")
    candidates.sort(key=lambda mesh: len(mesh.data.polygons), reverse=True)
    primary = candidates[0]
    if len(candidates) > 1:
        runner_up_faces = len(candidates[1].data.polygons)
        if len(primary.data.polygons) < max(1000, 10 * runner_up_faces):
            raise RuntimeError("I23D GLB has ambiguous textured mesh candidates")
    return primary


def apply_direct_bind_decimation(target):
    before = {
        "vertices": len(target.data.vertices),
        "faces": len(target.data.polygons),
        "uv_layers": len(target.data.uv_layers),
        "material_slots": len(target.material_slots),
        "material_names": [
            slot.material.name if slot.material is not None else None
            for slot in target.material_slots
        ],
    }
    if before["faces"] > DIRECT_BIND_FACE_BUDGET:
        direct.select_only(target)
        modifier = target.modifiers.new(name="I23D Direct Bind Budget", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = DIRECT_BIND_FACE_BUDGET / float(before["faces"])
        modifier.use_collapse_triangulate = True
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        if "FINISHED" not in result:
            raise RuntimeError("could not apply I23D direct-bind face budget")
    after = {
        "vertices": len(target.data.vertices),
        "faces": len(target.data.polygons),
        "uv_layers": len(target.data.uv_layers),
        "material_slots": len(target.material_slots),
        "material_names": [
            slot.material.name if slot.material is not None else None
            for slot in target.material_slots
        ],
    }
    if after["faces"] > DIRECT_BIND_FACE_BUDGET:
        raise RuntimeError("I23D direct-bind face budget was not enforced")
    if after["uv_layers"] != before["uv_layers"] or after["uv_layers"] < 1:
        raise RuntimeError("I23D decimation changed the UV-layer contract")
    if (
        after["material_slots"] != before["material_slots"]
        or after["material_names"] != before["material_names"]
    ):
        raise RuntimeError("I23D decimation changed the material contract")
    if any(len(polygon.vertices) != 3 for polygon in target.data.polygons):
        raise RuntimeError("I23D direct-bind target must remain triangulated")
    return {"face_budget": DIRECT_BIND_FACE_BUDGET, "before": before, "after": after}


def validate_i23d_pbr_material(target):
    if len(target.material_slots) != 1 or target.material_slots[0].material is None:
        raise RuntimeError("I23D direct-bind target must keep one PBR material")
    material = target.material_slots[0].material
    if not material.use_nodes or material.node_tree is None:
        raise RuntimeError("I23D direct-bind material has no node tree")
    principled = [
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    ]
    if len(principled) != 1:
        raise RuntimeError("I23D direct-bind material must contain one Principled BSDF")
    shader = principled[0]
    for input_name in ("Base Color", "Metallic", "Roughness"):
        if len(shader.inputs[input_name].links) != 1:
            raise RuntimeError(f"I23D PBR input is not texture-linked: {input_name}")
    images = []
    for node in material_image_nodes(material):
        image = node.image
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise RuntimeError("I23D PBR image has invalid dimensions")
        if image.packed_file is None:
            image.pack()
        if image.packed_file is None or image.packed_file.size <= 0:
            raise RuntimeError("I23D PBR image was not packed into bound.blend")
        images.append(
            {
                "node": node.name,
                "image": image.name,
                "pixel_size": [int(image.size[0]), int(image.size[1])],
                "packed_size_bytes": int(image.packed_file.size),
                "colorspace": image.colorspace_settings.name,
            }
        )
    if len(images) < 2:
        raise RuntimeError("I23D direct-bind material lost Base Color or packed PBR")
    return {
        "material_name": material.name,
        "material_slot_count": len(target.material_slots),
        "uv_layer_count": len(target.data.uv_layers),
        "images": images,
        "complete": True,
    }


def import_i23d_target(path, args, armature, source):
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError("could not import authenticated I23D GLB")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    target = select_primary_textured_mesh(meshes)
    canonical_yaw = math.pi if args.front_axis == "positive-y" else 0.0
    canonical_world = Matrix.Rotation(canonical_yaw, 4, "Z") @ target.matrix_world
    target.parent = None
    target.matrix_world = canonical_world
    for imported_object in imported:
        if imported_object != target:
            bpy.data.objects.remove(imported_object, do_unlink=True)
    target.name = "I23D_Rocketbox_Direct_Body"
    bpy.context.view_layer.update()

    if any(len(polygon.vertices) != 3 for polygon in target.data.polygons):
        direct.triangulate_mesh(target)
    if len(target.data.uv_layers) < 1 or len(target.material_slots) != 1:
        raise RuntimeError("I23D target lost its UV or material during import")
    raw_vertices = np.asarray(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float64,
    )
    decimation = apply_direct_bind_decimation(target)
    direct.move_target_to_armature_space(target, armature)
    target_regions, cleanup = direct.cleanup_target_geometry(target, source)
    pbr = validate_i23d_pbr_material(target)
    source_front = np.array(
        (0.0, -1.0 if args.front_axis == "negative-y" else 1.0, 0.0),
        dtype=np.float64,
    )
    yaw_basis = np.asarray(Matrix.Rotation(canonical_yaw, 3, "Z"), dtype=np.float64)
    canonical_front = yaw_basis @ source_front
    if not np.allclose(canonical_front, (0.0, -1.0, 0.0), atol=1.0e-7):
        raise RuntimeError("I23D front-axis normalization failed")
    metrics = {
        "backend": args.guide_backend,
        "axis_contract": {
            "source_front_axis": args.front_axis,
            "source_up_axis": "positive-z-after-gltf-import",
            "canonical_front_axis": "negative-y",
            "canonical_up_axis": "positive-z",
            "canonical_yaw_deg": 180.0 if canonical_yaw else 0.0,
            "canonical_front_vector": canonical_front.tolist(),
            "basis_determinant": float(np.linalg.det(yaw_basis)),
            "raw_extents_after_gltf_import": np.ptp(raw_vertices, axis=0).tolist(),
        },
        "discarded_import_objects": len(imported) - 1,
        "decimation": decimation,
        "cleanup": cleanup,
        "pbr": pbr,
    }
    return target, target_regions, metrics


def build_bind_manifest(
    args,
    output_dir,
    action_metrics,
    source_hashes,
    current_hashes,
    floor_z_m,
    consumed_inputs,
    axis_contract,
    guide_provenance,
):
    return {
        "schema_version": "hy3d_rocketbox_bind_v1",
        "asset_id": args.asset_id,
        "binding_mode": BINDING_MODE,
        "usage_scope": guide_provenance["usage_scope"],
        "guide_backend": args.guide_backend,
        "research_release_ok": guide_provenance["research_release_ok"],
        "permissive_commercial_ok": guide_provenance["permissive_commercial_ok"],
        "guide_provenance": guide_provenance,
        "floor_z_m": floor_z_m,
        "reference": direct.file_descriptor(output_dir / "reference.png"),
        "glbs": {
            "walk": direct.file_descriptor(output_dir / "bound_walk.glb"),
            "idle": direct.file_descriptor(output_dir / "bound_idle.glb"),
        },
        "bound_blend": direct.file_descriptor(output_dir / "bound.blend"),
        "cleaned_obj_contract": {
            "role": "direct_i23d_geometry_only",
            "materials": False,
            "uv": True,
            "normals": True,
        },
        "action_names": {
            "walk": action_metrics["walk"]["action_name"],
            "idle": action_metrics["idle"]["action_name"],
        },
        "artifacts": {
            "cleaned_obj": direct.file_descriptor(output_dir / "cleaned.obj"),
            "bound_blend": direct.file_descriptor(output_dir / "bound.blend"),
            "bound_walk_glb": direct.file_descriptor(output_dir / "bound_walk.glb"),
            "bound_idle_glb": direct.file_descriptor(output_dir / "bound_idle.glb"),
            "bind_metrics": direct.file_descriptor(output_dir / "bind_metrics.json"),
        },
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": consumed_inputs,
        "axis_contract": axis_contract,
    }


def build_direct_binding(
    args,
    output_dir,
    baseline,
    guide_input,
    idle,
    source_hashes,
    snapshot,
):
    result = bpy.ops.wm.open_mainfile(filepath=str(snapshot["paths"]["baseline_blend"]))
    if "FINISHED" not in result:
        raise RuntimeError("could not open immutable Rocketbox baseline")
    direct.retarget.configure_animation_scene()
    armature, source_mesh = direct.identify_target_objects()
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("approved Rocketbox walk action is missing")
    walk_action = armature.animation_data.action
    if walk_action.name != f"{args.asset_id}_walk_neutral_retarget":
        raise RuntimeError("approved Rocketbox walk action changed")
    walk_action.use_fake_user = True
    source = direct.capture_rocketbox_source(armature, source_mesh)
    floor_z_m = float(source["floor_z_m"])

    imported_target = import_i23d_target(
        snapshot["paths"]["i23d_guide_glb"], args, armature, source
    )
    runtime_mesh = imported_target[0]
    target_regions = imported_target[1]
    import_metrics = imported_target[2]
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    cleaned_obj = direct.export_cleaned_obj(runtime_mesh, output_dir / "cleaned.obj")
    pbr_before_bind = validate_i23d_pbr_material(runtime_mesh)
    binding = direct.bind_target_mesh(
        runtime_mesh, armature, source, target_regions
    )
    direct.remove_original_body(source_mesh)
    direct.validate_target_only_scene(armature, runtime_mesh)

    idle_action, idle_bake = direct.bake_idle_action(
        armature, args.asset_id, snapshot["paths"]["idle_motion_fbx"]
    )
    idle_action.use_fake_user = True
    action_set = direct.validate_two_actions(walk_action, idle_action)
    floor_normalization = {}
    for motion, action in (("walk", walk_action), ("idle", idle_action)):
        try:
            floor_normalization[motion] = template_fit.normalize_action_floor_contact(
                armature, runtime_mesh, action, floor_z_m
            )
        except RuntimeError:
            foot_indices = template_fit.foot_weighted_vertex_indices(runtime_mesh)
            failure = {
                "motion": motion,
                "floor_z_m": floor_z_m,
                "foot_weighted_vertex_counts": {
                    side: len(indices) for side, indices in foot_indices.items()
                },
                "state_after_failure": template_fit.sample_floor_state(
                    armature, runtime_mesh, action, foot_indices
                ),
            }
            direct.atomic_write_json(output_dir / "floor_failure.json", failure)
            print("I23D_DIRECT_FLOOR_FAILURE " + json.dumps(failure, sort_keys=True))
            raise
    action_metrics = {
        "walk": {
            "action_name": walk_action.name,
            "frame_start": direct.action_frame_range(walk_action)[0],
            "frame_end": direct.action_frame_range(walk_action)[1],
            "source": "approved Rocketbox neutral walk",
        },
        "idle": {
            "action_name": idle_action.name,
            "frame_start": direct.action_frame_range(idle_action)[0],
            "frame_end": direct.action_frame_range(idle_action)[1],
            "source": "gender-matched Rocketbox neutral idle",
        },
    }
    direct.validate_bound_weights(runtime_mesh)
    validate_i23d_pbr_material(runtime_mesh)
    direct.save_bound_blend(armature, runtime_mesh, output_dir / "bound.blend")

    walk_name = walk_action.name
    idle_name = idle_action.name
    armature, runtime_mesh, walk_action, idle_action = direct.load_saved_target(
        output_dir / "bound.blend", walk_name, idle_name
    )
    bound_pbr = validate_i23d_pbr_material(runtime_mesh)
    direct.validate_bound_weights(runtime_mesh)
    direct.validate_two_actions(walk_action, idle_action)
    expected_mesh = direct.retarget.mesh_metrics(runtime_mesh, armature)
    walk_skin = template_fit.capture_action_skin_contract(
        armature, runtime_mesh, walk_action
    )
    walk_start, walk_end, walk_positions = direct.sample_action_positions(
        armature, walk_action
    )
    idle_skin = template_fit.capture_action_skin_contract(
        armature, runtime_mesh, idle_action
    )
    idle_start, idle_end, idle_positions = direct.sample_action_positions(
        armature, idle_action
    )

    direct.isolate_action_for_export(armature, walk_action)
    direct.export_single_action_glb(
        armature, runtime_mesh, walk_action, output_dir / "bound_walk.glb"
    )
    armature, runtime_mesh, walk_action, idle_action = direct.load_saved_target(
        output_dir / "bound.blend", walk_name, idle_name
    )
    direct.isolate_action_for_export(armature, idle_action)
    direct.export_single_action_glb(
        armature, runtime_mesh, idle_action, output_dir / "bound_idle.glb"
    )
    glb_structure = {
        "walk": direct.inspect_bound_glb(output_dir / "bound_walk.glb"),
        "idle": direct.inspect_bound_glb(output_dir / "bound_idle.glb"),
    }
    direct.atomic_copy(snapshot["paths"]["i23d_reference"], output_dir / "reference.png")
    glb_roundtrip = {
        "walk": direct.retarget.roundtrip_validate(
            output_dir / "bound_walk.glb",
            expected_mesh,
            walk_positions,
            walk_skin,
            walk_start,
            walk_end,
        ),
        "idle": direct.retarget.roundtrip_validate(
            output_dir / "bound_idle.glb",
            expected_mesh,
            idle_positions,
            idle_skin,
            idle_start,
            idle_end,
        ),
    }
    for motion, values in glb_roundtrip.items():
        if not values["skin_weight_validation"]["passed"]:
            raise RuntimeError(f"{motion} direct-bind GLB skin validation failed")
        if values["maximum_world_joint_error_m"] >= values["joint_tolerance_m"]:
            raise RuntimeError(f"{motion} direct-bind GLB joint validation failed")

    current_hashes = template_fit.verify_i23d_source_hashes_current(
        baseline, guide_input, idle, source_hashes
    )
    guide_provenance = guide_input["provenance"]
    bind_metrics = {
        "schema_version": "i23d_rocketbox_bind_metrics_v1",
        "asset_id": args.asset_id,
        "binding_mode": BINDING_MODE,
        "usage_scope": guide_provenance["usage_scope"],
        "guide_provenance": guide_provenance,
        "floor_z_m": floor_z_m,
        "axis_contract": import_metrics["axis_contract"],
        "source_capture": {
            "mesh_name": source["mesh_name"],
            "vertex_count": source["vertex_count"],
            "face_count": source["face_count"],
            "bone_count": source["bone_count"],
        },
        "direct_target": import_metrics,
        "pbr_before_bind": pbr_before_bind,
        "bound_pbr": bound_pbr,
        "cleaned_obj": cleaned_obj,
        "binding": binding,
        "floor_normalization": floor_normalization,
        "actions": action_metrics,
        "bound_action_set": action_set,
        "idle_bake": idle_bake,
        "glb_structure": glb_structure,
        "glb_roundtrip": glb_roundtrip,
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": snapshot["records"],
        "outputs": list(OUTPUT_FILENAMES),
    }
    direct.atomic_write_json(output_dir / "bind_metrics.json", bind_metrics)
    manifest = build_bind_manifest(
        args,
        output_dir,
        action_metrics,
        source_hashes,
        current_hashes,
        floor_z_m,
        snapshot["records"],
        import_metrics["axis_contract"],
        guide_provenance,
    )
    direct.atomic_write_json(output_dir / "bind_manifest.json", manifest)
    print(
        "I23D_ROCKETBOX_DIRECT_BIND_OK "
        f"asset_id={args.asset_id} backend={args.guide_backend}"
    )
    return manifest


def invalidate_outputs(output_dir):
    direct.invalidate_readiness(output_dir)
    for filename in OUTPUT_FILENAMES:
        try:
            (output_dir / filename).unlink()
        except FileNotFoundError:
            pass
    try:
        (output_dir / "floor_failure.json").unlink()
    except FileNotFoundError:
        pass


def run_direct_binding(args):
    baseline = direct.validate_baseline_inputs(args)
    guide_input = validate_i23d_inputs(args)
    idle = direct.validate_idle_motion(args)
    source_hashes = template_fit.capture_i23d_source_hashes(
        baseline, guide_input, idle
    )
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    invalidate_outputs(output_dir)
    snapshot = template_fit.stage_i23d_input_snapshot(
        output_dir, baseline, guide_input, idle
    )
    try:
        return build_direct_binding(
            args,
            output_dir,
            baseline,
            guide_input,
            idle,
            source_hashes,
            snapshot,
        )
    finally:
        shutil.rmtree(snapshot["root"], ignore_errors=True)


def main(argv=None):
    run_direct_binding(parse_args(argv))


if __name__ == "__main__":
    main()
