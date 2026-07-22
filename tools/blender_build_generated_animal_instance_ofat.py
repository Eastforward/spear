#!/usr/bin/env python3
"""Build nine research OFAT instances from one accepted generated animal.

Geometry variation is deliberately separate from appearance variation. Size,
body build and life stage are bounded transforms of the accepted generated
mesh/rig. Coat variants must arrive as independently generated, real-reference
FLUX multiview edits projected back onto the exact same mesh and UVs. This tool
never implements a coat by multiplying an RGB material factor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

import bpy
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_build_stable_quadruped_instance as stable  # noqa: E402
from tools import blender_build_stable_animal_instance as textured  # noqa: E402
from tools import blender_robust_swap_mesh_keep_rig as robust  # noqa: E402


SCHEMA = "avengine_generated_animal_instance_ofat_v2"
SIZE_RATIOS = {"small": 0.85, "medium": 1.0, "large": 1.15}
BUILD_RATIOS = {"slim": 0.84, "standard": 1.0, "stocky": 1.16}
HEAD_RATIOS = {"young": 1.12, "adult": 1.0, "senior": 0.97}
SENIOR_MUZZLE_GRAY_MIX = 0.55
SENIOR_MUZZLE_GRAY_FLOOR = 0.68
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")
GROUND_TOLERANCE_M = 1.0e-6
MUZZLE_FORWARD_QUANTILE = 0.82


def parse_key_paths(values, label):
    result = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or IDENTIFIER.fullmatch(key) is None or key in result:
            raise RuntimeError(f"invalid or duplicate {label}: {value}")
        result[key] = Path(raw_path).resolve()
    return result


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--breed", required=True)
    parser.add_argument("--baseline-coat", required=True)
    parser.add_argument(
        "--coat-glb",
        action="append",
        default=[],
        metavar="COAT_ID=GLB",
        help="Exactly three breed-scoped coat GLBs, including the baseline.",
    )
    parser.add_argument(
        "--coat-projection-manifest",
        action="append",
        default=[],
        metavar="COAT_ID=MANIFEST",
        help="FLUX multiview projection evidence for each non-baseline coat.",
    )
    parser.add_argument(
        "--baseline-generation-manifest",
        type=Path,
        required=True,
        help="Real-reference FLUX generation evidence for the accepted source coat.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or unsafe {label}: {path}")
    return path


def load_json(path: Path, label: str):
    path = require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace output root: {path}")
    path.mkdir(parents=True)
    return path


def primary_skinned_mesh():
    candidates = [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH"
        and item.vertex_groups
        and any(modifier.type == "ARMATURE" for modifier in item.modifiers)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one generated skinned mesh, got {len(candidates)}")
    return candidates[0]


def smoothstep(values):
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def shape_weights(coordinates):
    semantic = robust.dog_region_coords(coordinates, "positive-x")
    minimum = semantic.min(axis=0)
    extent = np.maximum(semantic.max(axis=0) - minimum, 1.0e-12)
    normalized = (semantic - minimum) / extent
    forward = normalized[:, 0]
    height = normalized[:, 1]
    head = smoothstep((forward - 0.53) / 0.20) * smoothstep(
        (height - 0.28) / 0.22
    )
    central_forward = smoothstep((forward - 0.14) / 0.18) * (
        1.0 - smoothstep((forward - 0.58) / 0.16)
    )
    above_legs = smoothstep((height - 0.24) / 0.22)
    torso = central_forward * above_legs * (1.0 - head)
    if int(np.count_nonzero(head > 0.25)) < 20:
        raise RuntimeError("generated animal head region is too small")
    if int(np.count_nonzero(torso > 0.25)) < 20:
        raise RuntimeError("generated animal torso region is too small")
    return head, torso


def weighted_center(coordinates, weights):
    total = float(weights.sum())
    if total <= 1.0e-8:
        raise RuntimeError("shape-control weights have zero mass")
    return np.sum(coordinates * weights[:, None], axis=0) / total


def weighted_rms_radius(coordinates, weights, axes):
    center = weighted_center(coordinates, weights)
    selected = list(axes)
    offsets = coordinates[:, selected] - center[selected]
    return float(
        np.sqrt(
            np.sum(weights * np.sum(np.square(offsets), axis=1))
            / float(weights.sum())
        )
    )


def apply_shape(mesh, base_coordinates, body_build, life_stage):
    coordinates = base_coordinates.copy()
    head, torso = shape_weights(coordinates)
    torso_rms_before = weighted_rms_radius(coordinates, torso, (1, 2))
    head_rms_before = weighted_rms_radius(coordinates, head, (0, 1, 2))
    torso_center = weighted_center(coordinates, torso)
    head_center = weighted_center(coordinates, head)
    girth = BUILD_RATIOS[body_build]
    lateral = torso_center[1] + (coordinates[:, 1] - torso_center[1]) * girth
    vertical_scale = 1.0 + (girth - 1.0) * 0.45
    vertical = torso_center[2] + (coordinates[:, 2] - torso_center[2]) * vertical_scale
    coordinates[:, 1] += (lateral - coordinates[:, 1]) * torso
    coordinates[:, 2] += (vertical - coordinates[:, 2]) * torso
    head_scale = HEAD_RATIOS[life_stage]
    scaled_head = head_center + (coordinates - head_center) * head_scale
    coordinates += (scaled_head - coordinates) * head[:, None]
    torso_rms_after = weighted_rms_radius(coordinates, torso, (1, 2))
    head_rms_after = weighted_rms_radius(coordinates, head, (0, 1, 2))
    mesh.data.vertices.foreach_set("co", coordinates.reshape(-1))
    mesh.data.update()
    return {
        "torso_vertices_over_0_25": int(np.count_nonzero(torso > 0.25)),
        "head_vertices_over_0_25": int(np.count_nonzero(head > 0.25)),
        "torso_girth_scale": girth,
        "torso_vertical_scale": vertical_scale,
        "head_scale": head_scale,
        "soft_geometric_transition": True,
        "coat_pixels_modified": False,
        "semantic_measurements": {
            "torso_weighted_lateral_vertical_rms_before": torso_rms_before,
            "torso_weighted_lateral_vertical_rms_after": torso_rms_after,
            "torso_weighted_lateral_vertical_rms_ratio": (
                torso_rms_after / max(torso_rms_before, 1.0e-12)
            ),
            "head_weighted_radius_rms_before": head_rms_before,
            "head_weighted_radius_rms_after": head_rms_after,
            "head_weighted_radius_rms_ratio": (
                head_rms_after / max(head_rms_before, 1.0e-12)
            ),
        },
    }, head, coordinates


def apply_senior_muzzle_surface_cue(mesh, head, coordinates, output_texture):
    """Add a local senior muzzle cue without redefining the breed coat.

    The UV mask is derived from the accepted generated mesh's semantic head
    region.  Dark muzzle pixels are lifted toward neutral grey; already-light
    fur keeps its measured luminance, so a white blaze is never darkened into
    an artificial painted patch.  This is spatial age evidence, not a global
    material factor and not one of the three FLUX-authored coat identities.
    """

    node = textured.color_image_node(mesh)
    source = node.image
    width, height = map(int, source.size)
    if width <= 0 or height <= 0:
        raise RuntimeError("senior source Base Color is not loaded")
    pixels = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(pixels)
    pixels = pixels.reshape((height, width, 4))
    rgb = pixels[:, :, :3]
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    mask, mask_record = textured.rasterize_muzzle_mask(
        mesh, head, coordinates, width, height
    )
    target_luminance = np.maximum(luminance, SENIOR_MUZZLE_GRAY_FLOOR)
    neutral = np.repeat(target_luminance[:, :, None], 3, axis=2)
    alpha = np.clip(mask * SENIOR_MUZZLE_GRAY_MIX, 0.0, 1.0)[:, :, None]
    rgb[:] = np.clip(rgb * (1.0 - alpha) + neutral * alpha, 0.0, 1.0)

    realized = source.copy()
    realized.name = "GeneratedAnimalSeniorBaseColor"
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
        "method": "semantic_uv_muzzle_neutral_gray_floor_v1",
        "age_surface_pixels_modified": True,
        "global_rgb_material_factor_used": False,
        "muzzle_gray_mix": SENIOR_MUZZLE_GRAY_MIX,
        "muzzle_gray_luminance_floor": SENIOR_MUZZLE_GRAY_FLOOR,
        "already_light_fur_luminance_preserved": True,
        "output_texture": str(output_texture.resolve()),
        **mask_record,
    }


def rest_world_coordinates(mesh, armature):
    """Return undeformed mesh coordinates in the exported asset-root frame."""

    previous_pose_position = armature.data.pose_position
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    try:
        matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
        local = stable.mesh_coordinates(mesh)
        homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float64)))
        return (homogeneous @ matrix.T)[:, :3]
    finally:
        armature.data.pose_position = previous_pose_position
        bpy.context.view_layer.update()


def ground_instance_root(mesh, armature, root):
    """Translate the instance root so the rest-mesh sole minimum is exactly Z=0."""

    before = rest_world_coordinates(mesh, armature)
    minimum_before = float(before[:, 2].min())
    root.location.z -= minimum_before
    bpy.context.view_layer.update()
    after = rest_world_coordinates(mesh, armature)
    minimum_after = float(after[:, 2].min())
    if abs(minimum_after) > GROUND_TOLERANCE_M:
        raise RuntimeError(
            f"generated instance grounding readback failed: min_z={minimum_after}"
        )
    return {
        "method": "rest_mesh_minimum_z_to_asset_root_zero_v1",
        "rest_minimum_z_before_m": minimum_before,
        "root_translation_z_m": float(root.location.z),
        "rest_minimum_z_after_m": minimum_after,
        "tolerance_m": GROUND_TOLERANCE_M,
        "passed": True,
    }


def derive_muzzle_emitter(mesh, armature):
    """Derive a simple fixed mouth emitter from this concrete asset's rest mesh."""

    world = rest_world_coordinates(mesh, armature)
    semantic = robust.dog_region_coords(world, "positive-x")
    head, _torso = shape_weights(world)
    candidates = np.flatnonzero(head > 0.25)
    if len(candidates) < 20:
        raise RuntimeError("generated animal lacks enough head vertices for emitter")
    threshold = float(
        np.quantile(semantic[candidates, 0], MUZZLE_FORWARD_QUANTILE)
    )
    muzzle = candidates[semantic[candidates, 0] >= threshold]
    if len(muzzle) < 4:
        raise RuntimeError("generated animal lacks enough forward muzzle vertices")
    weights = np.maximum(head[muzzle], 1.0e-6)
    point = weighted_center(semantic[muzzle], weights)
    # A fixed emitter should stay on the animal's sagittal plane instead of
    # inheriting an arbitrary left/right surface sample from an asymmetric mesh.
    point[2] = 0.0
    return {
        "method": "semantic_head_forward_quantile_rest_mesh_v1",
        "coordinate_system": "avengine_local_x_forward_y_up_z_left_m",
        "emitter_offset_m": [float(value) for value in point],
        "local_forward_axis": [1.0, 0.0, 0.0],
        "muzzle_forward_quantile": MUZZLE_FORWARD_QUANTILE,
        "candidate_vertex_count": int(len(candidates)),
        "selected_vertex_count": int(len(muzzle)),
        "asset_specific_not_species_template": True,
        "mouth_animation_required": False,
    }


def rounded_digest(records):
    digest = hashlib.sha256()
    for value in records:
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        else:
            digest.update(struct.pack("<d", round(float(value), 6)))
    return digest.hexdigest()


def asset_authority_signature(mesh, armature):
    coordinates = stable.mesh_coordinates(mesh)
    topology = []
    for polygon in mesh.data.polygons:
        topology.extend((len(polygon.vertices), *polygon.vertices))
    uv = []
    if mesh.data.uv_layers.active is None:
        raise RuntimeError("generated animal has no active UV layer")
    for item in mesh.data.uv_layers.active.data:
        uv.extend(item.uv)
    weights = []
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    for vertex in mesh.data.vertices:
        weights.append(str(vertex.index))
        for membership in sorted(vertex.groups, key=lambda item: group_names[item.group]):
            weights.extend((group_names[membership.group], membership.weight))
    bones = []
    for bone in sorted(armature.data.bones, key=lambda item: item.name):
        bones.extend(
            (
                bone.name,
                bone.parent.name if bone.parent else "",
                *bone.head_local,
                *bone.tail_local,
            )
        )
    return {
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "loop_count": len(mesh.data.loops),
        "bone_count": len(armature.data.bones),
        "rest_position_digest_rounded_1e-6": rounded_digest(coordinates.reshape(-1)),
        "topology_digest": rounded_digest(topology),
        "uv_digest_rounded_1e-6": rounded_digest(uv),
        "skin_weights_digest_rounded_1e-6": rounded_digest(weights),
        "rest_bones_digest_rounded_1e-6": rounded_digest(bones),
        "idle_walking_keyframe_digest": stable.action_sha256(list(bpy.data.actions)),
    }


def import_asset(path: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(path))
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one fitted armature, got {len(armatures)}")
    armature = armatures[0]
    mesh = primary_skinned_mesh()
    actions = stable.canonical_actions(armature)
    if sorted(action.name for action in actions) != ["Idle", "Walking"]:
        raise RuntimeError("generated asset must contain exact Idle and Walking actions")
    return mesh, armature


def glb_chunks(path: Path):
    payload = path.read_bytes()
    if len(payload) < 20:
        raise RuntimeError(f"GLB is truncated: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise RuntimeError(f"invalid GLB header: {path}")
    chunks = []
    offset = 12
    while offset < len(payload):
        length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunks.append((chunk_type, payload[offset : offset + length]))
        offset += length
    if offset != len(payload) or not chunks or chunks[0][0] != 0x4E4F534A:
        raise RuntimeError(f"invalid GLB chunks: {path}")
    return chunks


def output_readback(path: Path):
    document = json.loads(glb_chunks(path)[0][1].decode("utf-8"))
    animations = sorted(item.get("name") for item in document.get("animations", []))
    primitives = [
        primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    skinned = [
        item
        for item in primitives
        if {"JOINTS_0", "WEIGHTS_0"}.issubset(item.get("attributes", {}))
    ]
    if animations != ["Idle", "Walking"]:
        raise RuntimeError(f"Walk/Idle readback failed: {animations}")
    if len(skinned) != 1 or len(document.get("skins", [])) != 1:
        raise RuntimeError("generated instance lost its single skinned mesh")
    materials = document.get("materials", [])
    if not materials or any(
        "baseColorTexture"
        not in material.get("pbrMetallicRoughness", {})
        for material in materials
    ):
        raise RuntimeError("generated instance lost its Base Color texture")
    return {
        "animations": animations,
        "skin_count": len(document.get("skins", [])),
        "skinned_primitive_count": len(skinned),
        "base_color_texture_material_count": len(materials),
    }


def validate_baseline_generation(path: Path, baseline_coat: str):
    value = load_json(path, "baseline FLUX generation manifest")
    board = value.get("inputs", {}).get("appearance_reference_board") or value.get(
        "appearance_reference_board"
    )
    board_path = board.get("path") if isinstance(board, dict) else board
    model = value.get("model", {})
    if (
        not board_path
        or not Path(board_path).resolve().is_file()
        or not isinstance(model, dict)
        or model.get("is_distilled") is not False
    ):
        raise RuntimeError("baseline coat lacks real-reference undistilled FLUX evidence")
    return {
        "coat_id": baseline_coat,
        "method": "real_reference_flux_source_asset_generation",
        "manifest": str(path.resolve()),
        "appearance_reference_board": str(Path(board_path).resolve()),
        "not_global_rgb_factor": True,
    }


def validate_projection_evidence(coat_id, manifest_path, coat_glb, input_glb):
    projection = load_json(manifest_path, f"{coat_id} coat projection manifest")
    if (
        projection.get("schema") != "avengine_generated_animal_multiview_coat_projection_v2"
        or projection.get("not_global_rgb_factor") is not True
        or projection.get("geometry_skin_skeleton_and_actions_preserved_by_design") is not True
        or Path(projection.get("input_glb", "")).resolve() != input_glb
        or Path(projection.get("output_glb", "")).resolve() != coat_glb
    ):
        raise RuntimeError(f"{coat_id} projection does not authenticate this asset")
    flux_manifest_path = Path(projection["edited_view_dir"]).resolve().parent / "manifest.json"
    flux = load_json(flux_manifest_path, f"{coat_id} FLUX edit manifest")
    if (
        not str(flux.get("schema", "")).startswith("avengine_flux2_base_animal_multiview_coat_edit_v")
        or flux.get("is_distilled") is not False
        or flux.get("one_model_invocation") is not True
        or flux.get("reference_image_count") != 2
        or not flux.get("appearance_reference_board")
        or flux.get("geometry_rig_or_animation_edit_authorized") is not False
    ):
        raise RuntimeError(f"{coat_id} lacks real-reference undistilled FLUX evidence")
    return {
        "coat_id": coat_id,
        "method": "real_reference_flux_multiview_edit_then_uv_projection",
        "projection_manifest": str(manifest_path.resolve()),
        "flux_edit_manifest": str(flux_manifest_path),
        "appearance_reference_board": flux["appearance_reference_board"],
        "not_global_rgb_factor": True,
    }


def variant_matrix(coat_ids, baseline_coat):
    base = {
        "size": "medium",
        "body_build": "standard",
        "coat": baseline_coat,
        "life_stage": "adult",
    }
    return [
        ("baseline", dict(base), None),
        ("size_small", {**base, "size": "small"}, "size"),
        ("size_large", {**base, "size": "large"}, "size"),
        ("build_slim", {**base, "body_build": "slim"}, "body_build"),
        ("build_stocky", {**base, "body_build": "stocky"}, "body_build"),
        *[
            (f"coat_{coat}", {**base, "coat": coat}, "coat")
            for coat in sorted(coat_ids)
            if coat != baseline_coat
        ],
        ("age_young", {**base, "life_stage": "young"}, "life_stage"),
        ("age_senior", {**base, "life_stage": "senior"}, "life_stage"),
    ]


def write_json_exclusive(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    args = parse_argv()
    if IDENTIFIER.fullmatch(args.breed) is None or IDENTIFIER.fullmatch(args.baseline_coat) is None:
        raise RuntimeError("breed and baseline coat must be portable lowercase identifiers")
    input_glb = require_file(args.input_glb, "accepted generated animal GLB")
    coat_glbs = parse_key_paths(args.coat_glb, "coat GLB")
    projection_manifests = parse_key_paths(
        args.coat_projection_manifest, "coat projection manifest"
    )
    if len(coat_glbs) != 3 or args.baseline_coat not in coat_glbs:
        raise RuntimeError("exactly three breed-scoped coat GLBs are required")
    coat_glbs = {key: require_file(path, f"{key} coat GLB") for key, path in coat_glbs.items()}
    if coat_glbs[args.baseline_coat] != input_glb:
        raise RuntimeError("baseline coat GLB must be the accepted input GLB")
    nonbaseline = set(coat_glbs) - {args.baseline_coat}
    if set(projection_manifests) != nonbaseline:
        raise RuntimeError("each non-baseline coat needs exactly one projection manifest")
    output_root = require_new_directory(args.output_root)

    baseline_evidence = validate_baseline_generation(
        args.baseline_generation_manifest.resolve(), args.baseline_coat
    )
    coat_evidence = {args.baseline_coat: baseline_evidence}
    for coat_id in sorted(nonbaseline):
        coat_evidence[coat_id] = validate_projection_evidence(
            coat_id,
            projection_manifests[coat_id],
            coat_glbs[coat_id],
            input_glb,
        )

    baseline_mesh, baseline_armature = import_asset(input_glb)
    baseline_signature = asset_authority_signature(baseline_mesh, baseline_armature)
    for coat_id in sorted(nonbaseline):
        coat_mesh, coat_armature = import_asset(coat_glbs[coat_id])
        signature = asset_authority_signature(coat_mesh, coat_armature)
        if signature != baseline_signature:
            differences = {
                key: {
                    "baseline": baseline_signature.get(key),
                    "coat": signature.get(key),
                }
                for key in sorted(set(baseline_signature) | set(signature))
                if baseline_signature.get(key) != signature.get(key)
            }
            raise RuntimeError(
                f"{coat_id} coat edit changed protected asset authority: "
                f"{json.dumps(differences, sort_keys=True)}"
            )
        coat_evidence[coat_id]["authority_signature_matches_baseline"] = True

    results = []
    for variant_id, attributes, changed_attribute in variant_matrix(
        coat_glbs, args.baseline_coat
    ):
        variant_root = output_root / variant_id
        variant_root.mkdir()
        output = variant_root / "instance.glb"
        if changed_attribute == "coat":
            mesh, armature = import_asset(coat_glbs[attributes["coat"]])
            shape = {
                "torso_girth_scale": 1.0,
                "torso_vertical_scale": 1.0,
                "head_scale": 1.0,
                "coat_pixels_modified": True,
            }
            life_stage_surface = {
                "method": "none",
                "age_surface_pixels_modified": False,
            }
            source_path = coat_glbs[attributes["coat"]]
        else:
            mesh, armature = import_asset(input_glb)
            base_coordinates = stable.mesh_coordinates(mesh)
            shape, head, shaped_coordinates = apply_shape(
                mesh,
                base_coordinates,
                attributes["body_build"],
                attributes["life_stage"],
            )
            if attributes["life_stage"] == "senior":
                life_stage_surface = apply_senior_muzzle_surface_cue(
                    mesh,
                    head,
                    shaped_coordinates,
                    variant_root / "senior_base_color.png",
                )
            else:
                life_stage_surface = {
                    "method": "none",
                    "age_surface_pixels_modified": False,
                }
            source_path = input_glb
        root = stable.install_instance_scale(
            list(bpy.data.objects), SIZE_RATIOS[attributes["size"]], 0
        )
        grounding = ground_instance_root(mesh, armature, root)
        emitter = derive_muzzle_emitter(mesh, armature)
        stable.export_instance(output)
        readback = output_readback(output)
        record = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "variant_id": variant_id,
            "changed_attribute_from_baseline": changed_attribute,
            "source_glb": str(source_path),
            "source_sha256": sha256_file(source_path),
            "output_glb": str(output.resolve()),
            "output_sha256": sha256_file(output),
            "attributes": {"breed": args.breed, **attributes},
            "size_ratio": SIZE_RATIOS[attributes["size"]],
            "shape": shape,
            "life_stage_surface": life_stage_surface,
            "grounding": grounding,
            "emitter_anchor": emitter,
            "appearance": coat_evidence[attributes["coat"]],
            "readback": readback,
            "actual_generated_mesh_preserved": True,
            "template_geometry_used": False,
            "global_rgb_material_factor_used": False,
            "walk_idle_preserved": True,
        }
        write_json_exclusive(variant_root / "manifest.json", record)
        results.append(record)

    expected = 1 + 2 + 2 + 2 + 2
    if len(results) != expected:
        raise RuntimeError(f"expected nine OFAT variants, got {len(results)}")
    batch = {
        "schema": "avengine_generated_animal_instance_ofat_batch_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "breed": args.breed,
        "baseline_input_glb": str(input_glb),
        "baseline_authority_signature": baseline_signature,
        "coat_ids": sorted(coat_glbs),
        "attribute_domains": {
            "size": list(SIZE_RATIOS),
            "body_build": list(BUILD_RATIOS),
            "life_stage": list(HEAD_RATIOS),
            "coat": sorted(coat_glbs),
        },
        "variant_count": len(results),
        "results": [
            {
                "variant_id": item["variant_id"],
                "changed_attribute_from_baseline": item[
                    "changed_attribute_from_baseline"
                ],
                "attributes": item["attributes"],
                "output_glb": item["output_glb"],
                "output_sha256": item["output_sha256"],
                "readback": item["readback"],
            }
            for item in results
        ],
    }
    write_json_exclusive(output_root / "batch_manifest.json", batch)
    print(f"GENERATED_ANIMAL_INSTANCE_OFAT_OK variants={len(results)} output={output_root}")


if __name__ == "__main__":
    main()
