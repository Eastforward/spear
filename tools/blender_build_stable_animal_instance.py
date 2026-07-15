#!/usr/bin/env python3
"""Realize one authenticated stable-animal instance without re-rigging it."""

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

import bpy
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_robust_swap_mesh_keep_rig as robust  # noqa: E402
from tools import prepare_controlled_source_asset_execution as preflight_lib  # noqa: E402


SCHEMA = "avengine_stable_animal_instance_realization_v1"


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
        consumers = job.get("consumer_requests", [])
        if any(item.get("instance_id") == instance_id for item in consumers):
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
    required = {"size", "body_build", "coat_tone", "life_stage"}
    if set(operations) != required:
        raise RuntimeError(
            f"stable beagle instance controls must be exactly {sorted(required)}"
        )
    return operations


def skin_uv_topology_sha256(mesh) -> str:
    digest = hashlib.sha256()
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


def vertex_position_sha256(mesh) -> str:
    digest = hashlib.sha256()
    for vertex in mesh.data.vertices:
        digest.update(struct.pack("<3d", *map(float, vertex.co)))
    return digest.hexdigest()


def action_sha256(actions) -> str:
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


def parse_optional_csv(value, label):
    if value is None:
        return ()
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a CSV string")
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise RuntimeError(f"{label} cannot be empty")
    return result


def vertex_group_weight(mesh, vertex, tokens, exact_group_names=()):
    names = {group.index: group.name.lower() for group in mesh.vertex_groups}
    exact = set(exact_group_names)
    return min(
        1.0,
        sum(
            float(membership.weight)
            for membership in vertex.groups
            if (
                names.get(membership.group, "") in exact
                if exact
                else any(
                    token in names.get(membership.group, "") for token in tokens
                )
            )
        ),
    )


def weighted_rms_radius(coordinates, weights, axes):
    total = float(weights.sum())
    center = np.sum(coordinates * weights[:, None], axis=0) / total
    selected_axes = list(axes)
    offsets = coordinates[:, selected_axes] - center[selected_axes]
    return float(
        np.sqrt(np.sum(weights * np.sum(np.square(offsets), axis=1)) / total)
    )


def apply_shape_controls(
    mesh,
    torso_girth_scale,
    head_scale,
    *,
    torso_group_names=(),
    head_group_names=(),
):
    count = len(mesh.data.vertices)
    coordinates = np.empty(count * 3, dtype=np.float64)
    mesh.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape((-1, 3))
    torso = np.asarray(
        [
            vertex_group_weight(
                mesh,
                vertex,
                ("pelvis", "spine"),
                exact_group_names=torso_group_names,
            )
            for vertex in mesh.data.vertices
        ],
        dtype=np.float64,
    )
    head = np.asarray(
        [
            vertex_group_weight(
                mesh,
                vertex,
                ("head", "ear", "mouth", "eye"),
                exact_group_names=head_group_names,
            )
            for vertex in mesh.data.vertices
        ],
        dtype=np.float64,
    )
    if np.count_nonzero(torso > 0.05) < 100 or np.count_nonzero(head > 0.05) < 50:
        raise RuntimeError("stable template lacks usable torso/head semantic skin groups")

    torso_rms_before = weighted_rms_radius(coordinates, torso, (1, 2))
    head_rms_before = weighted_rms_radius(coordinates, head, (0, 1, 2))
    torso_weight_sum = float(torso.sum())
    torso_center_y = float(np.sum(coordinates[:, 1] * torso) / torso_weight_sum)
    torso_center_z = float(np.sum(coordinates[:, 2] * torso) / torso_weight_sum)
    torso_y_scale = 1.0 + (float(torso_girth_scale) - 1.0) * torso
    torso_z_scale = 1.0 + (float(torso_girth_scale) - 1.0) * 0.55 * torso
    coordinates[:, 1] = torso_center_y + (
        coordinates[:, 1] - torso_center_y
    ) * torso_y_scale
    coordinates[:, 2] = torso_center_z + (
        coordinates[:, 2] - torso_center_z
    ) * torso_z_scale

    head_weight_sum = float(head.sum())
    head_center = np.sum(coordinates * head[:, None], axis=0) / head_weight_sum
    head_factor = 1.0 + (float(head_scale) - 1.0) * head
    coordinates = head_center + (coordinates - head_center) * head_factor[:, None]
    torso_rms_after = weighted_rms_radius(coordinates, torso, (1, 2))
    head_rms_after = weighted_rms_radius(coordinates, head, (0, 1, 2))
    mesh.data.vertices.foreach_set("co", coordinates.reshape(-1))
    mesh.data.update()
    return {
        "torso_selected_vertices": int(np.count_nonzero(torso > 0.05)),
        "head_selected_vertices": int(np.count_nonzero(head > 0.05)),
        "torso_girth_scale": float(torso_girth_scale),
        "head_scale": float(head_scale),
        "torso_group_names_csv": ",".join(torso_group_names),
        "head_group_names_csv": ",".join(head_group_names),
        "semantic_measurements": {
            "torso_weighted_lateral_rms_before": torso_rms_before,
            "torso_weighted_lateral_rms_after": torso_rms_after,
            "torso_weighted_lateral_rms_ratio": torso_rms_after
            / max(torso_rms_before, 1.0e-12),
            "head_weighted_radius_rms_before": head_rms_before,
            "head_weighted_radius_rms_after": head_rms_after,
            "head_weighted_radius_rms_ratio": head_rms_after
            / max(head_rms_before, 1.0e-12),
        },
    }, head, coordinates


def color_image_node(mesh):
    for material in mesh.data.materials:
        if not material or not material.use_nodes:
            continue
        node = material.node_tree.nodes.get("Rocketbox Color")
        if node is not None and getattr(node, "image", None) is not None:
            return node
        for candidate in material.node_tree.nodes:
            if (
                candidate.type == "TEX_IMAGE"
                and candidate.image is not None
                and candidate.image.colorspace_settings.name == "sRGB"
            ):
                return candidate
    raise RuntimeError("stable template has no identifiable base-color image")


def rasterize_muzzle_mask(mesh, head_weights, coordinates, width, height):
    head_indices = np.flatnonzero(head_weights > 0.05)
    head_x = coordinates[head_indices, 0]
    threshold = float(np.quantile(head_x, 0.58))
    maximum = float(head_x.max())
    span = max(maximum - threshold, 1.0e-9)
    forward = np.clip((coordinates[:, 0] - threshold) / span, 0.0, 1.0)
    vertex_mask = np.clip(head_weights * forward, 0.0, 1.0)
    mask = np.zeros((height, width), dtype=np.float32)
    uv_data = mesh.data.uv_layers.active.data
    for polygon in mesh.data.polygons:
        if len(polygon.loop_indices) != 3:
            continue
        vertices = list(polygon.vertices)
        weights = vertex_mask[vertices]
        if float(weights.max()) <= 0.01:
            continue
        uv = np.asarray(
            [uv_data[index].uv[:] for index in polygon.loop_indices],
            dtype=np.float64,
        )
        uv = np.clip(uv, 0.0, 1.0)
        points = np.column_stack(
            (uv[:, 0] * (width - 1), uv[:, 1] * (height - 1))
        )
        x0 = max(0, int(math.floor(points[:, 0].min())))
        x1 = min(width - 1, int(math.ceil(points[:, 0].max())))
        y0 = max(0, int(math.floor(points[:, 1].min())))
        y1 = min(height - 1, int(math.ceil(points[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        denominator = (
            (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (points[0, 1] - points[2, 1])
        )
        if abs(denominator) < 1.0e-8:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        bary0 = (
            (points[1, 1] - points[2, 1]) * (xx - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (yy - points[2, 1])
        ) / denominator
        bary1 = (
            (points[2, 1] - points[0, 1]) * (xx - points[2, 0])
            + (points[0, 0] - points[2, 0]) * (yy - points[2, 1])
        ) / denominator
        bary2 = 1.0 - bary0 - bary1
        inside = (bary0 >= -1.0e-5) & (bary1 >= -1.0e-5) & (bary2 >= -1.0e-5)
        interpolated = bary0 * weights[0] + bary1 * weights[1] + bary2 * weights[2]
        region = mask[y0 : y1 + 1, x0 : x1 + 1]
        np.maximum(region, np.where(inside, interpolated, 0.0), out=region)
    return np.clip(mask, 0.0, 1.0), {
        "muzzle_forward_quantile": 0.58,
        "muzzle_mask_nonzero_pixels": int(np.count_nonzero(mask > 0.01)),
        "muzzle_mask_max": float(mask.max()),
    }


def realize_texture(
    mesh,
    output_texture,
    coat_gain,
    muzzle_gray_mix,
    muzzle_gray_target,
    coat_desaturation,
    head,
    coordinates,
):
    node = color_image_node(mesh)
    source = node.image
    width, height = map(int, source.size)
    if width <= 0 or height <= 0:
        raise RuntimeError("base-color image is not loaded")
    pixels = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(pixels)
    pixels = pixels.reshape((height, width, 4))
    rgb = pixels[:, :, :3]
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-6)
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    white_coat = (luminance > 0.52) & (saturation < 0.20)
    pigmented_coat = (~white_coat) & (maximum > 0.01)
    if not np.any(pigmented_coat):
        raise RuntimeError("base-color image has no measurable nonwhite coat pixels")
    mean_luminance_before = float(np.mean(luminance[pigmented_coat]))
    coat_strength = pigmented_coat.astype(np.float32)
    factor = 1.0 + (float(coat_gain) - 1.0) * coat_strength[:, :, None]
    rgb[:] = np.clip(rgb * factor, 0.0, 1.0)
    if not 0.0 <= float(coat_desaturation) <= 1.0:
        raise RuntimeError("coat_desaturation must be in [0, 1]")
    if float(coat_desaturation) > 0.0:
        adjusted_luminance = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
        desaturated = np.repeat(adjusted_luminance[:, :, None], 3, axis=2)
        mix = pigmented_coat[:, :, None].astype(np.float32) * float(
            coat_desaturation
        )
        rgb[:] = np.clip(rgb * (1.0 - mix) + desaturated * mix, 0.0, 1.0)

    muzzle_mask, mask_record = rasterize_muzzle_mask(
        mesh, head, coordinates, width, height
    )
    if not 0.0 <= float(muzzle_gray_target) <= 1.0:
        raise RuntimeError("muzzle_gray_target must be in [0, 1]")
    if float(muzzle_gray_mix) > 0.0:
        gray = np.full(rgb.shape[:2], float(muzzle_gray_target), dtype=np.float32)
        alpha = np.clip(
            muzzle_mask * float(muzzle_gray_mix), 0.0, 1.0
        )[:, :, None]
        rgb[:] = rgb * (1.0 - alpha) + gray[:, :, None] * alpha

    final_luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    mean_luminance_after = float(np.mean(final_luminance[pigmented_coat]))

    realized = source.copy()
    realized.name = "StableAnimalInstanceBaseColor"
    realized.pixels.foreach_set(pixels.reshape(-1))
    realized.update()
    realized.filepath_raw = str(output_texture)
    realized.file_format = "PNG"
    realized.save()
    reloaded = bpy.data.images.load(str(output_texture), check_existing=False)
    reloaded.colorspace_settings.name = "sRGB"
    reloaded.pack()
    node.image = reloaded
    return {
        "source_image": source.name,
        "resolution": [width, height],
        "coat_luminance_gain": float(coat_gain),
        "coat_desaturation": float(coat_desaturation),
        "mean_nonwhite_coat_luminance_before": mean_luminance_before,
        "mean_nonwhite_coat_luminance_after": mean_luminance_after,
        "measured_nonwhite_coat_pixels": int(np.count_nonzero(pigmented_coat)),
        "muzzle_gray_mix": float(muzzle_gray_mix),
        "muzzle_gray_target": float(muzzle_gray_target),
        **mask_record,
    }


def install_instance_scale(armature, ratio):
    root = bpy.data.objects.new("StableAnimalInstanceScaleRoot", None)
    bpy.context.collection.objects.link(root)
    world = armature.matrix_world.copy()
    armature.parent = root
    armature.matrix_world = world
    root.scale = (float(ratio),) * 3
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


def export_instance(root, armature, mesh, output):
    bpy.ops.object.select_all(action="DESELECT")
    for item in (root, armature, mesh):
        item.select_set(True)
    bpy.context.view_layer.objects.active = armature
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
    texture_path = require_output(
        output_path.with_name(f"{output_path.stem}.base_color.png"),
        "realized base-color texture",
    )
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
        raise RuntimeError("stable template must contain one armature and a mesh")
    armature = armatures[0]
    mesh = max(meshes, key=lambda item: len(item.data.vertices))
    for item in list(bpy.data.objects):
        if item not in {armature, mesh}:
            bpy.data.objects.remove(item, do_unlink=True)

    actions = canonical_actions(armature)
    contract_before = skin_uv_topology_sha256(mesh)
    action_contract_before = action_sha256(actions)
    position_before = vertex_position_sha256(mesh)
    body_parameters = operations["body_build"]["parameters"]
    age_parameters = operations["life_stage"]["parameters"]
    shape_record, head, coordinates = apply_shape_controls(
        mesh,
        body_parameters["torso_girth_scale"],
        age_parameters["head_scale"],
        torso_group_names=parse_optional_csv(
            body_parameters.get("torso_group_names_csv"),
            "torso_group_names_csv",
        ),
        head_group_names=parse_optional_csv(
            age_parameters.get("head_group_names_csv"),
            "head_group_names_csv",
        ),
    )
    texture_record = realize_texture(
        mesh,
        texture_path,
        operations["coat_tone"]["parameters"]["coat_luminance_gain"],
        age_parameters["muzzle_gray_mix"],
        age_parameters.get("muzzle_gray_target", 0.90),
        age_parameters.get("coat_desaturation", 0.0),
        head,
        coordinates,
    )
    scale_ratio = float(job["target_physical_profile"]["scale_ratio"])
    root = install_instance_scale(armature, scale_ratio)
    contract_after = skin_uv_topology_sha256(mesh)
    action_contract_after = action_sha256(actions)
    position_after = vertex_position_sha256(mesh)
    if contract_before != contract_after:
        raise RuntimeError("instance realization changed topology, UVs, or skin weights")
    if action_contract_before != action_contract_after:
        raise RuntimeError("instance realization changed Walk/Idle keyframes")
    export_instance(root, armature, mesh, output_path)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "instance_id": args.instance_id,
        "execution_job_id": job["execution_job_id"],
        "request": job["consumer_requests"][0],
        "taxonomy": job["taxonomy"],
        "fixed_attributes": job["fixed_attributes"],
        "sampled_attributes": job["sampled_attributes"],
        "appearance_reference": job["stable_instance_plan"]["appearance_reference"],
        "attribute_operations": job["stable_instance_plan"]["attribute_operations"],
        "acoustic_profile": job["acoustic_profile"],
        "target_physical_profile": job["target_physical_profile"],
        "realization": {
            "builder": "stable_animal_textured_pbr_v2",
            "uniform_instance_scale": scale_ratio,
            "runtime_front_axis": "positive_x",
            "automatic_fine_yaw_inference": False,
            "shape": shape_record,
            "texture": texture_record,
            "topology_uv_skin_sha256_before": contract_before,
            "topology_uv_skin_sha256_after": contract_after,
            "topology_uv_skin_unchanged": True,
            "action_sha256_before": action_contract_before,
            "action_sha256_after": action_contract_after,
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
            },
            "base_color": {
                "path": str(texture_path),
                "sha256": sha256_file(texture_path),
                "size_bytes": texture_path.stat().st_size,
            },
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
        "STABLE_ANIMAL_INSTANCE_OK "
        f"instance={args.instance_id} output={output_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
