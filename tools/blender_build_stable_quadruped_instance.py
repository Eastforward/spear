#!/usr/bin/env python3
"""Realize one low-poly native quadruped instance without changing its rig.

This is the solid-material companion to ``blender_build_stable_animal_instance``.
It supports the audited Quaternius templates, whose body axis and material
layout differ from the textured Beagle template.  Geometry edits are driven by
semantic skin groups; coat edits touch only profile-declared PBR materials.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Iterable, Sequence

import bpy
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_robust_swap_mesh_keep_rig as robust  # noqa: E402
from tools import prepare_controlled_source_asset_execution as preflight_lib  # noqa: E402


SCHEMA = "avengine_stable_quadruped_instance_realization_v1"
REQUIRED_ATTRIBUTES = {"size", "body_build", "coat_tone", "life_stage"}


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or unsafe {label}: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_job(preflight_path: Path, instance_id: str):
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload = preflight_lib.validate_execution_preflight(payload)
    matches = []
    for job in payload["routes"]["stable_animal_template_v1"]:
        if any(
            request.get("instance_id") == instance_id
            for request in job.get("consumer_requests", [])
        ):
            matches.append(job)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one stable job for {instance_id!r}, got {len(matches)}"
        )
    return payload, matches[0]


def selected_operations(job):
    operations = {
        item["attribute"]: item
        for item in job["stable_instance_plan"]["attribute_operations"]
    }
    if set(operations) != REQUIRED_ATTRIBUTES:
        raise RuntimeError(
            f"stable quadruped controls must be exactly {sorted(REQUIRED_ATTRIBUTES)}"
        )
    coat = operations["coat_tone"]["parameters"]
    if coat.get("surface_mode") != "solid_material_pbr":
        raise RuntimeError("this builder requires solid_material_pbr")
    return operations


def parse_csv(value, label: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a CSV string")
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise RuntimeError(f"{label} cannot be empty")
    return result


def mesh_contract_sha256(meshes: Iterable[bpy.types.Object]) -> str:
    digest = hashlib.sha256()
    for mesh in sorted(meshes, key=lambda item: item.name):
        digest.update(mesh.name.encode("utf-8"))
        for polygon in mesh.data.polygons:
            digest.update(struct.pack("<I", len(polygon.vertices)))
            digest.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
        for layer in mesh.data.uv_layers:
            digest.update(layer.name.encode("utf-8"))
            for item in layer.data:
                digest.update(struct.pack("<2d", *map(float, item.uv)))
        for group in mesh.vertex_groups:
            digest.update(group.name.encode("utf-8"))
        for vertex in mesh.data.vertices:
            for membership in sorted(vertex.groups, key=lambda item: item.group):
                digest.update(
                    struct.pack("<Id", int(membership.group), float(membership.weight))
                )
    return digest.hexdigest()


def vertex_position_sha256(meshes: Iterable[bpy.types.Object]) -> str:
    digest = hashlib.sha256()
    for mesh in sorted(meshes, key=lambda item: item.name):
        digest.update(mesh.name.encode("utf-8"))
        for vertex in mesh.data.vertices:
            digest.update(struct.pack("<3d", *map(float, vertex.co)))
    return digest.hexdigest()


def action_sha256(actions: Iterable[bpy.types.Action]) -> str:
    digest = hashlib.sha256()
    for action in sorted(actions, key=lambda item: item.name):
        digest.update(action.name.encode("utf-8"))
        for curve in sorted(
            action.fcurves, key=lambda item: (item.data_path, item.array_index)
        ):
            digest.update(curve.data_path.encode("utf-8"))
            digest.update(struct.pack("<I", int(curve.array_index)))
            for point in curve.keyframe_points:
                digest.update(struct.pack("<2d", *map(float, point.co)))
    return digest.hexdigest()


def vertex_semantic_weights(mesh, tokens: Sequence[str]) -> np.ndarray:
    names = {group.index: group.name.lower() for group in mesh.vertex_groups}
    lowered = tuple(token.lower() for token in tokens)
    return np.asarray(
        [
            min(
                1.0,
                sum(
                    float(membership.weight)
                    for membership in vertex.groups
                    if any(token in names.get(membership.group, "") for token in lowered)
                ),
            )
            for vertex in mesh.data.vertices
        ],
        dtype=np.float64,
    )


def mesh_coordinates(mesh) -> np.ndarray:
    coordinates = np.empty(len(mesh.data.vertices) * 3, dtype=np.float64)
    mesh.data.vertices.foreach_get("co", coordinates)
    return coordinates.reshape((-1, 3))


def weighted_center(coordinates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 1.0e-8:
        raise RuntimeError("semantic weights have zero mass")
    return np.sum(coordinates * weights[:, None], axis=0) / total


def infer_body_frame(meshes, torso_tokens, head_tokens):
    candidates = []
    for mesh in meshes:
        coordinates = mesh_coordinates(mesh)
        torso = vertex_semantic_weights(mesh, torso_tokens)
        head = vertex_semantic_weights(mesh, head_tokens)
        torso_count = int(np.count_nonzero(torso > 0.05))
        head_count = int(np.count_nonzero(head > 0.05))
        if torso_count >= 12 and head_count >= 8:
            candidates.append(
                (
                    torso_count + head_count,
                    mesh,
                    coordinates,
                    torso,
                    head,
                )
            )
    if not candidates:
        raise RuntimeError("stable template lacks usable torso/head semantic skin groups")
    _, mesh, coordinates, torso, head = max(candidates, key=lambda item: item[0])
    torso_center = weighted_center(coordinates, torso)
    head_center = weighted_center(coordinates, head)
    forward = head_center[:2] - torso_center[:2]
    norm = float(np.linalg.norm(forward))
    if norm < 1.0e-5:
        centered = coordinates[:, :2] - torso_center[:2]
        covariance = (centered * torso[:, None]).T @ centered / max(float(torso.sum()), 1e-8)
        values, vectors = np.linalg.eigh(covariance)
        forward = vectors[:, int(np.argmax(values))]
        if float(np.dot(head_center[:2] - torso_center[:2], forward)) < 0.0:
            forward = -forward
        norm = float(np.linalg.norm(forward))
    forward /= max(norm, 1.0e-8)
    width = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    return {
        "authority_mesh": mesh.name,
        "torso_center": torso_center,
        "head_center": head_center,
        "forward_xy": forward,
        "width_xy": width,
    }


def apply_shape_controls(
    meshes,
    torso_tokens,
    head_tokens,
    torso_girth_scale,
    head_scale,
):
    frame = infer_body_frame(meshes, torso_tokens, head_tokens)
    forward = frame["forward_xy"]
    width = frame["width_xy"]
    records = []
    total_torso = 0
    total_head = 0
    for mesh in meshes:
        coordinates = mesh_coordinates(mesh)
        torso = vertex_semantic_weights(mesh, torso_tokens)
        head = vertex_semantic_weights(mesh, head_tokens)
        torso_count = int(np.count_nonzero(torso > 0.05))
        head_count = int(np.count_nonzero(head > 0.05))
        total_torso += torso_count
        total_head += head_count
        if float(torso.sum()) > 1.0e-8:
            center = weighted_center(coordinates, torso)
            centered_xy = coordinates[:, :2] - center[:2]
            longitudinal = centered_xy @ forward
            lateral = centered_xy @ width
            lateral *= 1.0 + (float(torso_girth_scale) - 1.0) * torso
            coordinates[:, :2] = (
                center[:2]
                + longitudinal[:, None] * forward[None, :]
                + lateral[:, None] * width[None, :]
            )
            height_scale = 1.0 + (float(torso_girth_scale) - 1.0) * 0.55 * torso
            coordinates[:, 2] = center[2] + (coordinates[:, 2] - center[2]) * height_scale
        if float(head.sum()) > 1.0e-8:
            center = weighted_center(coordinates, head)
            factor = 1.0 + (float(head_scale) - 1.0) * head
            coordinates = center + (coordinates - center) * factor[:, None]
        mesh.data.vertices.foreach_set("co", coordinates.reshape(-1))
        mesh.data.update()
        records.append(
            {
                "mesh": mesh.name,
                "vertices": len(mesh.data.vertices),
                "torso_selected_vertices": torso_count,
                "head_selected_vertices": head_count,
            }
        )
    if total_torso < 12 or total_head < 8:
        raise RuntimeError("stable template semantic vertex coverage is insufficient")
    return {
        "torso_group_tokens": list(torso_tokens),
        "head_group_tokens": list(head_tokens),
        "torso_girth_scale": float(torso_girth_scale),
        "head_scale": float(head_scale),
        "body_frame": {
            "authority_mesh": frame["authority_mesh"],
            "torso_center": list(map(float, frame["torso_center"])),
            "head_center": list(map(float, frame["head_center"])),
            "forward_xy": list(map(float, frame["forward_xy"])),
            "width_xy": list(map(float, frame["width_xy"])),
        },
        "meshes": records,
    }


def principled_base_color(material):
    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED" and "Base Color" in node.inputs:
                return node.inputs["Base Color"]
    return None


def get_material_rgba(material):
    socket = principled_base_color(material)
    if socket is not None and not socket.is_linked:
        return tuple(map(float, socket.default_value))
    return tuple(map(float, material.diffuse_color))


def set_material_rgba(material, rgba):
    rgba = tuple(float(value) for value in rgba)
    material.diffuse_color = rgba
    socket = principled_base_color(material)
    if socket is not None and not socket.is_linked:
        socket.default_value = rgba


def transform_rgb(rgb, gain, desaturation):
    values = np.asarray(rgb, dtype=np.float64)
    values = np.clip(values * float(gain), 0.0, 1.0)
    luminance = float(np.dot(values, np.asarray([0.2126, 0.7152, 0.0722])))
    mix = float(desaturation)
    return np.clip(values * (1.0 - mix) + luminance * mix, 0.0, 1.0)


def apply_material_controls(
    coat_names,
    muzzle_names,
    coat_gain,
    muzzle_gray_mix,
    muzzle_gray_target,
    senior_coat_desaturation,
):
    materials = {material.name: material for material in bpy.data.materials}
    missing_coat = sorted(set(coat_names) - set(materials))
    if missing_coat:
        raise RuntimeError(f"declared coat materials are missing: {missing_coat}")
    actual_muzzles = [name for name in muzzle_names if name.lower() != "none"]
    missing_muzzle = sorted(set(actual_muzzles) - set(materials))
    if missing_muzzle:
        raise RuntimeError(f"declared muzzle materials are missing: {missing_muzzle}")
    if not 0.0 <= float(muzzle_gray_mix) <= 1.0:
        raise RuntimeError("muzzle_gray_mix must be in [0, 1]")
    if not 0.0 <= float(muzzle_gray_target) <= 1.0:
        raise RuntimeError("muzzle_gray_target must be in [0, 1]")
    if not 0.0 <= float(senior_coat_desaturation) <= 1.0:
        raise RuntimeError("senior_coat_desaturation must be in [0, 1]")

    records = []
    for name in coat_names:
        material = materials[name]
        before = get_material_rgba(material)
        after_rgb = transform_rgb(
            before[:3], coat_gain, senior_coat_desaturation
        )
        after = (*map(float, after_rgb), before[3])
        set_material_rgba(material, after)
        records.append({"material": name, "role": "coat", "before": before, "after": after})
    for name in actual_muzzles:
        material = materials[name]
        before = get_material_rgba(material)
        target = np.full(3, float(muzzle_gray_target), dtype=np.float64)
        after_rgb = (
            np.asarray(before[:3]) * (1.0 - float(muzzle_gray_mix))
            + target * float(muzzle_gray_mix)
        )
        after = (*map(float, np.clip(after_rgb, 0.0, 1.0)), before[3])
        set_material_rgba(material, after)
        records.append({"material": name, "role": "muzzle", "before": before, "after": after})
    return {
        "surface_mode": "solid_material_pbr",
        "coat_luminance_gain": float(coat_gain),
        "senior_coat_desaturation": float(senior_coat_desaturation),
        "muzzle_gray_mix": float(muzzle_gray_mix),
        "muzzle_gray_target": float(muzzle_gray_target),
        "materials": records,
    }


def install_instance_scale(objects, ratio, cardinal_yaw_deg):
    if cardinal_yaw_deg not in {-90, 0, 90, 180}:
        raise RuntimeError("template cardinal yaw must be -90/0/90/180")
    root = bpy.data.objects.new("StableQuadrupedInstanceScaleRoot", None)
    bpy.context.collection.objects.link(root)
    for item in list(objects):
        if item is root or item.parent is not None:
            continue
        world = item.matrix_world.copy()
        item.parent = root
        item.matrix_world = world
    root.scale = (float(ratio),) * 3
    root.rotation_euler[2] = math.radians(float(cardinal_yaw_deg))
    bpy.context.view_layer.update()
    return root


def canonical_actions(armature):
    actions = list(bpy.data.actions)
    idle = next((item for item in actions if item.name.lower().startswith("idle")), None)
    walk = next(
        (item for item in actions if item.name.lower().startswith(("walk", "walking"))),
        None,
    )
    if idle is None or walk is None:
        raise RuntimeError(f"stable template lacks Walk/Idle: {[a.name for a in actions]}")
    armature.animation_data_create()
    armature.animation_data.action = None
    while armature.animation_data.nla_tracks:
        armature.animation_data.nla_tracks.remove(armature.animation_data.nla_tracks[0])
    for name, action in (("Walking", walk), ("Idle", idle)):
        action.name = name
        start, end = map(float, action.frame_range)
        track = armature.animation_data.nla_tracks.new()
        track.name = name
        strip = track.strips.new(name, int(round(start)), action)
        strip.action_frame_start = start
        strip.action_frame_end = end
    return [walk, idle]


def export_instance(output):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_nla_strips=True,
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    robust.postprocess_glb_animation_channels(
        output, {"translation", "rotation"}, canonical_walk_idle=True
    )


def main():
    args = parse_argv()
    preflight_path = require_input(args.preflight, "execution preflight")
    output_path = require_output(args.output_glb, "output GLB")
    manifest_path = require_output(args.manifest, "manifest")
    preflight, job = load_job(preflight_path, args.instance_id)
    operations = selected_operations(job)
    artifact = job["stable_instance_plan"]["base_template"]["artifact"]
    input_path = require_input(Path(artifact["resolved_path"]), "stable template")
    if (
        sha256_file(input_path) != artifact["sha256"]
        or input_path.stat().st_size != artifact["size_bytes"]
    ):
        raise RuntimeError("stable template no longer matches authenticated preflight")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise RuntimeError("stable template must contain one armature and one or more meshes")
    armature = armatures[0]
    skinned_meshes = [
        item
        for item in meshes
        if item.vertex_groups
        and any(modifier.type == "ARMATURE" for modifier in item.modifiers)
    ]
    if not skinned_meshes:
        raise RuntimeError("stable template has no skinned mesh")

    actions = canonical_actions(armature)
    contract_before = mesh_contract_sha256(meshes)
    action_before = action_sha256(actions)
    position_before = vertex_position_sha256(meshes)
    body = operations["body_build"]["parameters"]
    age = operations["life_stage"]["parameters"]
    coat = operations["coat_tone"]["parameters"]
    shape_record = apply_shape_controls(
        skinned_meshes,
        parse_csv(body["torso_group_tokens_csv"], "torso_group_tokens_csv"),
        parse_csv(age["head_group_tokens_csv"], "head_group_tokens_csv"),
        body["torso_girth_scale"],
        age["head_scale"],
    )
    material_record = apply_material_controls(
        parse_csv(coat["coat_material_names_csv"], "coat_material_names_csv"),
        parse_csv(age["muzzle_material_names_csv"], "muzzle_material_names_csv"),
        coat["coat_luminance_gain"],
        age["muzzle_gray_mix"],
        age["muzzle_gray_target"],
        age["senior_coat_desaturation"],
    )
    scale_ratio = float(job["target_physical_profile"]["scale_ratio"])
    cardinal_yaw = operations["size"]["parameters"]["template_cardinal_yaw_deg"]
    root = install_instance_scale(
        list(bpy.data.objects), scale_ratio, cardinal_yaw
    )
    contract_after = mesh_contract_sha256(meshes)
    action_after = action_sha256(actions)
    position_after = vertex_position_sha256(meshes)
    if contract_before != contract_after:
        raise RuntimeError("instance realization changed topology, UVs, or skin weights")
    if action_before != action_after:
        raise RuntimeError("instance realization changed Walk/Idle keyframes")
    export_instance(output_path)

    request = next(
        item
        for item in job["consumer_requests"]
        if item["instance_id"] == args.instance_id
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "instance_id": args.instance_id,
        "execution_job_id": job["execution_job_id"],
        "request": request,
        "taxonomy": job["taxonomy"],
        "fixed_attributes": job["fixed_attributes"],
        "sampled_attributes": job["sampled_attributes"],
        "appearance_reference": job["stable_instance_plan"]["appearance_reference"],
        "attribute_operations": job["stable_instance_plan"]["attribute_operations"],
        "acoustic_profile": job["acoustic_profile"],
        "target_physical_profile": job["target_physical_profile"],
        "realization": {
            "builder": "stable_quadruped_solid_material_v1",
            "uniform_instance_scale": scale_ratio,
            "template_cardinal_yaw_deg": cardinal_yaw,
            "runtime_front_axis": "positive_x",
            "automatic_fine_yaw_inference": False,
            "shape": shape_record,
            "materials": material_record,
            "scale_root": root.name,
            "mesh_count": len(meshes),
            "skinned_mesh_count": len(skinned_meshes),
            "topology_uv_skin_sha256_before": contract_before,
            "topology_uv_skin_sha256_after": contract_after,
            "topology_uv_skin_unchanged": True,
            "action_sha256_before": action_before,
            "action_sha256_after": action_after,
            "actions_unchanged": True,
            "vertex_position_sha256_before": position_before,
            "vertex_position_sha256_after": position_after,
        },
        "source_template": {
            "path": str(input_path),
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "preflight_sha256": preflight["preflight_sha256"],
        },
        "artifacts": {
            "glb": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        },
        "qa": {
            "glb_readback": "pending",
            "walking_deformation": "pending",
            "idle_deformation": "pending",
            "visual_attribute_realization": "pending",
            "ue_apartment": "pending",
            "audio": "pending",
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "STABLE_QUADRUPED_INSTANCE_OK "
        f"instance={args.instance_id} output={output_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
