"""Render sampled frames from a rigged GLB animation in Blender.

Usage:
  blender --background --python tools/blender_render_glb_animation.py -- \\
    --input tmp/hy3d/swap_test/Dog_robust_swap.glb \\
    --action Walking \\
    --output-dir /tmp/anim_robust_swap \\
    --n-frames 12
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import mathutils


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAR_ROOT = os.path.dirname(TOOLS_DIR)
if SPEAR_ROOT not in sys.path:
    sys.path.insert(0, SPEAR_ROOT)

from tools.generated_quadruped_semantics import (  # noqa: E402
    SemanticRigError,
    infer_quadruped_semantics,
)


MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO = 0.35


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--action", default="Walking")
    p.add_argument(
        "--rest-pose",
        action="store_true",
        help="Render the authored armature rest pose instead of an animation action.",
    )
    p.add_argument(
        "--quadruped-far-limb-offset-ratio",
        type=float,
        default=0.0,
        help=(
            "For a canonicalized quadruped authored rest pose (head +X, up +Z), "
            "translate only the geometry-inferred far-side front/hind limb "
            "chains by this fraction of the hip span so four grounded limbs "
            "remain visible from an exact side camera."
        ),
    )
    p.add_argument(
        "--quadruped-far-limb-action-pose-ratio",
        type=float,
        default=0.0,
        help=(
            "For a canonicalized quadruped authored rest pose, compose the "
            "far-side fore/hind chains from independently selected grounded "
            "frames of a native action. The target fore-aft separation is this "
            "fraction of the hip span. This preserves native joint rotations "
            "and avoids translating a shoulder/hip chain through the skin."
        ),
    )
    p.add_argument(
        "--quadruped-pose-action",
        default="Walking",
        help="Native action used to find grounded far-limb chain poses.",
    )
    p.add_argument(
        "--quadruped-pose-samples",
        type=int,
        default=57,
        help="Number of native-action frames sampled for each far limb.",
    )
    p.add_argument(
        "--pose-template-clay-color",
        default=None,
        help=(
            "Optional #RRGGBB uniform clay material for a rest-pose image "
            "template. This also enables smooth polygon shading so source "
            "material patches and low-poly facets cannot leak into FLUX.2."
        ),
    )
    p.add_argument(
        "--pose-template-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "For an orthographic rest-pose template only, yaw the complete "
            "animal by at most 30 degrees to expose near/far limb depth. "
            "This is source-view evidence, never automatic direction inference."
        ),
    )
    p.add_argument(
        "--review-clay-color",
        default=None,
        help=(
            "Optional #RRGGBB material override for animation QA. This is "
            "render-only evidence for source GLBs whose vertex-color alpha "
            "material is not visible in headless Eevee; it never edits or "
            "exports the input asset."
        ),
    )
    p.add_argument(
        "--preserve-volume",
        action="store_true",
        help=(
            "Render-only dual-quaternion armature deformation diagnostic. "
            "This never edits or exports the input GLB."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--view", default="side", choices=["side", "front", "quarter"])
    p.add_argument(
        "--asset-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "Manual cardinal rotation of the complete rigged asset around Blender "
            "+Z. Only -90/0/90/180 are accepted; this is used to build reviewer "
            "direction candidates and never performs automatic inference."
        ),
    )
    p.add_argument(
        "--trajectory-distance-ratio",
        type=float,
        default=0.0,
        help=(
            "Translate the complete asset from left to right along world +X by "
            "this multiple of its rest-pose diagonal while sampling the action."
        ),
    )
    p.add_argument(
        "--orthographic",
        action="store_true",
        help="Use an orthographic camera for canonical pose-template evidence.",
    )
    p.add_argument(
        "--camera-distance-multiplier",
        type=float,
        default=2.0,
        help="Camera distance as a multiple of the rest-pose mesh diagonal.",
    )
    p.add_argument(
        "--camera-reference-diagonal",
        type=float,
        default=0.0,
        help=(
            "Optional shared physical diagonal used only for camera framing. "
            "With --orthographic this keeps small/medium/large instances on "
            "one visual scale instead of auto-zooming every asset to fill the frame."
        ),
    )
    p.add_argument(
        "--ground-plane",
        action="store_true",
        help="Render a neutral floor at the rest-pose mesh minimum Z for foot-contact review.",
    )
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   choices=["BLENDER_EEVEE_NEXT", "CYCLES"])
    return p.parse_args(argv)


def look_at(obj, target):
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def choose_action(action_name):
    actions = list(bpy.data.actions)
    for action in actions:
        if action.name == action_name:
            return action
    lowered = action_name.lower()
    for action in actions:
        if lowered in action.name.lower():
            return action
    raise SystemExit(f"missing action {action_name}; available={[a.name for a in actions]}")


def mesh_bbox(meshes):
    mn = [1e9, 1e9, 1e9]
    mx = [-1e9, -1e9, -1e9]
    for obj in meshes:
        for vertex in obj.data.vertices:
            p = obj.matrix_world @ vertex.co
            for axis in range(3):
                mn[axis] = min(mn[axis], p[axis])
                mx[axis] = max(mx[axis], p[axis])
    return mn, mx


def add_review_ground(center, ground_z, diag):
    bpy.ops.mesh.primitive_plane_add(
        size=max(2.0, diag * 4.0),
        location=(center[0], center[1], ground_z - diag * 0.002),
    )
    ground = bpy.context.object
    ground.name = "FootContactReviewGround"
    material = bpy.data.materials.new(name="FootContactReviewGroundMaterial")
    material.diffuse_color = (0.18, 0.20, 0.22, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.18, 0.20, 0.22, 1.0)
    principled.inputs["Roughness"].default_value = 0.85
    ground.data.materials.append(material)
    print(f"[ground] z={ground.location.z:.6f} size={max(2.0, diag * 4.0):.3f}", flush=True)
    return ground


def parse_hex_color(value):
    if (
        not isinstance(value, str)
        or len(value) != 7
        or not value.startswith("#")
    ):
        raise SystemExit("--pose-template-clay-color must use #RRGGBB")
    try:
        channels = tuple(int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as error:
        raise SystemExit("--pose-template-clay-color must use #RRGGBB") from error
    return (*channels, 1.0)


def apply_pose_template_clay_material(body, color):
    rgba = parse_hex_color(color)
    material = bpy.data.materials.new(name="PoseTemplateUniformClay")
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = 0.78
    body.data.materials.clear()
    body.data.materials.append(material)
    for polygon in body.data.polygons:
        polygon.use_smooth = True
    print(
        f"[pose-template] template_material=uniform_clay color={color.lower()} "
        "smooth_shading=true",
        flush=True,
    )


def weighted_skin_bone_names(body, armature, minimum_weight=1.0e-8):
    """Return deform bones and ancestors, excluding detached IK controls."""
    group_names = {group.index: group.name for group in body.vertex_groups}
    weighted = set()
    weight_totals = {}
    for vertex in body.data.vertices:
        for membership in vertex.groups:
            if membership.weight <= minimum_weight:
                continue
            name = group_names.get(membership.group)
            if name is not None and armature.data.bones.get(name) is not None:
                weighted.add(name)
                weight_totals[name] = weight_totals.get(name, 0.0) + float(
                    membership.weight
                )
    if not weighted:
        raise SystemExit("skinned quadruped has no positive-weight bone groups")
    included = set(weighted)
    for name in tuple(weighted):
        bone = armature.data.bones.get(name)
        while bone is not None:
            included.add(bone.name)
            bone = bone.parent
    # Exporters commonly leave the terminal paw/hoof marker unweighted. Keep a
    # direct leaf child of a deform chain so geometry inference sees the actual
    # ground endpoint, while detached multi-bone Pole/IK control chains remain
    # excluded.
    terminal_markers = {
        child.name
        for name in tuple(included)
        for child in armature.data.bones[name].children
        if not child.children
    }
    included.update(terminal_markers)
    roots = sorted(
        name
        for name in included
        if armature.data.bones[name].parent is None
        or armature.data.bones[name].parent.name not in included
    )
    if len(roots) > 1:
        components = {}
        for root in roots:
            members = {root}
            members.update(
                child.name
                for child in armature.data.bones[root].children_recursive
                if child.name in included
            )
            components[root] = members
        chosen = max(
            roots,
            key=lambda root: (
                len(components[root]),
                sum(weight_totals.get(name, 0.0) for name in components[root]),
                root,
            ),
        )
        ignored = sorted(root for root in roots if root != chosen)
        included = components[chosen]
        print(
            f"[pose-template] deform_component_root={chosen} "
            f"deform_component_bones={len(included)} "
            f"terminal_markers={len(terminal_markers & included)} "
            f"ignored_detached_control_roots={ignored}",
            flush=True,
        )
    return included


def bone_records(armature, include_names=None):
    include_names = (
        set(include_names)
        if include_names is not None
        else {bone.name for bone in armature.data.bones}
    )
    records = []
    for bone in armature.data.bones:
        if bone.name not in include_names:
            continue
        records.append(
            {
                "name": bone.name,
                "parent": (
                    bone.parent.name
                    if bone.parent is not None and bone.parent.name in include_names
                    else None
                ),
                "children": sorted(
                    child.name for child in bone.children if child.name in include_names
                ),
                "head_world": list(armature.matrix_world @ bone.head_local),
                "tail_world": list(armature.matrix_world @ bone.tail_local),
            }
        )
    return records


def infer_canonical_quadruped(armature, body):
    minimum, maximum = mesh_bbox([body])
    extent = [maximum[index] - minimum[index] for index in range(3)]
    include_names = weighted_skin_bone_names(body, armature)
    records = bone_records(armature, include_names)
    try:
        semantics = infer_quadruped_semantics(
            records,
            bbox_min=minimum,
            bbox_extent=extent,
            front_axis="positive-x",
        )
    except SemanticRigError as error:
        raise SystemExit(f"quadruped far-limb semantic inference failed: {error}") from error
    return semantics, records, minimum, maximum, extent


def apply_quadruped_far_limb_offset(armature, body, offset_ratio):
    """Expose far-side limbs without yawing the torso or the camera.

    The caller first applies an explicit cardinal yaw so the animal faces world
    +X. Geometry/topology then identifies the four limb chains without relying
    on exporter-specific bone names. The positive-world-Y limbs are far from
    the renderer's side camera (located on -Y). Moving the far front chain
    slightly backward and the far hind chain slightly forward creates a compact
    neutral rigging stance rather than a walking stride. The translations have
    no world-Y/Z component: torso/head orientation and every paw's ground height
    remain unchanged.
    """
    semantics, records, _minimum, _maximum, _extent = infer_canonical_quadruped(
        armature, body
    )
    by_name = {record["name"]: record for record in records}
    far_front = semantics.front_side_positive[0]
    far_hind = semantics.hind_side_positive[0]
    hip_span = abs(
        by_name[far_front]["head_world"][0]
        - by_name[far_hind]["head_world"][0]
    )
    if hip_span <= 1.0e-9:
        raise SystemExit("quadruped far-limb front/hind span is degenerate")
    offset = hip_span * offset_ratio
    translations = {
        far_front: -offset,
        far_hind: offset,
    }
    world_to_armature = armature.matrix_world.inverted().to_3x3()
    for name, world_delta_x in translations.items():
        pose_bone = armature.pose.bones[name]
        local_delta = world_to_armature @ mathutils.Vector((world_delta_x, 0.0, 0.0))
        pose_bone.matrix = (
            mathutils.Matrix.Translation(local_delta) @ pose_bone.matrix
        )
        bpy.context.view_layer.update()
    print(
        f"[pose-template] far_limb_offset_ratio={offset_ratio:.4f} "
        f"hip_span={hip_span:.6f} offset={offset:.6f} "
        f"far_front_root={far_front} far_hind_root={far_hind} "
        "semantic_front_axis=world_positive_x "
        "torso_transform=identity camera_yaw_deg=0 ground_delta=0",
        flush=True,
    )


def set_scene_frame(scene, value):
    whole = math.floor(float(value))
    scene.frame_set(whole, subframe=float(value) - whole)


def pose_bone_world_head(armature, name):
    return armature.matrix_world @ armature.pose.bones[name].head


def reset_pose_basis(armature):
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis.identity()
    bpy.context.view_layer.update()


def apply_chain_basis(armature, chain, basis_by_name):
    for name in chain:
        armature.pose.bones[name].matrix_basis = basis_by_name[name].copy()
    bpy.context.view_layer.update()


def rotation_basis(quaternion, factor=1.0):
    """Return a translation-free native pose rotation blended from rest."""
    identity = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    rotation = identity.slerp(quaternion, float(factor))
    return rotation.to_matrix().to_4x4()


def blended_rotation_basis(rotations_by_name, factor):
    return {
        name: rotation_basis(rotation, factor)
        for name, rotation in rotations_by_name.items()
    }


def apply_quadruped_far_limb_action_pose(
    armature,
    body,
    target_ratio,
    action_name,
    sample_count,
):
    """Compose grounded far limbs from native action rotations.

    Each far limb is selected independently from the same native motion cycle.
    The final pose keeps the torso/head and both near-side limbs in authored rest
    pose. No chain root is translated by this function, so shoulder/hip skin
    remains connected and no rest-mesh vertex is edited.
    """
    semantics, records, _minimum, _maximum, extent = infer_canonical_quadruped(
        armature, body
    )
    by_name = {record["name"]: record for record in records}
    pairs = {
        "front": (
            semantics.front_side_negative,
            semantics.front_side_positive,
            -1.0,
        ),
        "hind": (
            semantics.hind_side_negative,
            semantics.hind_side_positive,
            1.0,
        ),
    }
    far_roots = [pair[1][0] for pair in pairs.values()]
    hip_span = abs(
        by_name[far_roots[0]]["head_world"][0]
        - by_name[far_roots[1]]["head_world"][0]
    )
    if hip_span <= 1.0e-9:
        raise SystemExit("quadruped action-pose front/hind span is degenerate")
    target_offset = hip_span * target_ratio

    reset_pose_basis(armature)
    base = {
        label: {
            "near": pose_bone_world_head(armature, near_chain[-1]),
            "far": pose_bone_world_head(armature, far_chain[-1]),
        }
        for label, (near_chain, far_chain, _direction) in pairs.items()
    }
    action = choose_action(action_name)
    start, end = action.frame_range
    if end <= start:
        raise SystemExit(f"quadruped pose action has no duration: {action.name}")
    armature.animation_data_create()
    armature.animation_data.action = action
    sampled = [
        start + (end - start) * index / max(1, sample_count - 1)
        for index in range(sample_count)
    ]
    action_rotations = []
    scene = bpy.context.scene
    for frame in sampled:
        set_scene_frame(scene, frame)
        bpy.context.view_layer.update()
        action_rotations.append(
            {
                "frame": float(frame),
                "rotations": {
                    name: armature.pose.bones[name].matrix_basis.to_quaternion()
                    for _label, (_near, far, _direction) in pairs.items()
                    for name in far
                },
            }
        )
    armature.animation_data_clear()

    chosen = {}
    height = max(float(extent[2]), 1.0e-9)
    for label, (_near_chain, far_chain, direction) in pairs.items():
        target_x = base[label]["near"].x + direction * target_offset
        target_z = base[label]["far"].z
        candidates = []
        # A full walking frame normally has one lifted hoof.  Interpolate only
        # native joint rotations back toward authored rest, then choose the
        # strongest separation that still satisfies the ground constraint.
        # This is not a translated limb or an invented IK target: every joint
        # stays on the native action arc and all matrix translations/scales are
        # deliberately discarded.
        blend_factors = [index / 32.0 for index in range(1, 33)]
        for sample in action_rotations:
            for factor in blend_factors:
                basis = blended_rotation_basis(sample["rotations"], factor)
                reset_pose_basis(armature)
                apply_chain_basis(armature, far_chain, basis)
                foot = pose_bone_world_head(armature, far_chain[-1])
                x_error = abs(foot.x - target_x) / hip_span
                z_error = abs(foot.z - target_z) / height
                separation = direction * (foot.x - base[label]["near"].x)
                direction_ok = separation > 0.0
                score = x_error + 6.0 * z_error + (
                    0.0 if direction_ok else 10.0
                )
                candidates.append(
                    {
                        "score": score,
                        "frame": sample["frame"],
                        "factor": factor,
                        "basis": basis,
                        "foot": foot.copy(),
                        "x_error": x_error,
                        "z_error": z_error,
                        "direction_ok": direction_ok,
                    }
                )
        valid = [
            candidate
            for candidate in candidates
            if candidate["direction_ok"] and candidate["z_error"] <= 0.025
        ]
        if not valid:
            best = min(candidates, key=lambda item: (item["score"], item["frame"]))
            raise SystemExit(
                "quadruped native action has no grounded separated "
                f"{label} far-limb pose; best_frame={best['frame']:.3f} "
                f"blend={best['factor']:.4f} "
                f"x_error={best['x_error']:.6f} z_error={best['z_error']:.6f}"
            )
        chosen[label] = min(valid, key=lambda item: (item["score"], item["frame"]))

    reset_pose_basis(armature)
    for label, (_near_chain, far_chain, _direction) in pairs.items():
        apply_chain_basis(armature, far_chain, chosen[label]["basis"])
    final_feet = {}
    for label, (near_chain, far_chain, _direction) in pairs.items():
        final_feet[f"{label}_near"] = pose_bone_world_head(armature, near_chain[-1])
        final_feet[f"{label}_far"] = pose_bone_world_head(armature, far_chain[-1])
    maximum_ground_delta = max(
        abs(final_feet[f"{label}_{side}"].z - base[label][side].z)
        for label in pairs
        for side in ("near", "far")
    )
    if maximum_ground_delta > 0.025 * height:
        raise SystemExit(
            "composed quadruped action pose changed ground height too much: "
            f"{maximum_ground_delta / height:.6f}"
        )
    print(
        f"[pose-template] far_limb_action_pose_ratio={target_ratio:.4f} "
        f"native_action={action.name} samples={sample_count} "
        f"front_frame={chosen['front']['frame']:.3f} "
        f"front_blend={chosen['front']['factor']:.4f} "
        f"hind_frame={chosen['hind']['frame']:.3f} "
        f"hind_blend={chosen['hind']['factor']:.4f} "
        f"front_x_error={chosen['front']['x_error']:.6f} "
        f"hind_x_error={chosen['hind']['x_error']:.6f} "
        f"maximum_ground_delta_ratio={maximum_ground_delta / height:.6f} "
        "chain_root_translation=false torso_transform=identity "
        "camera_yaw_deg=0 automatic_direction_inference=false",
        flush=True,
    )


def apply_asset_cardinal_yaw(armature, yaw_deg):
    """Rotate the whole skinned asset without changing any bone animation."""
    pivot = armature.matrix_world.translation.copy()
    asset_root = bpy.data.objects.new("ManualCardinalAssetRoot", None)
    bpy.context.collection.objects.link(asset_root)
    asset_root.location = pivot
    original_world = armature.matrix_world.copy()
    armature.parent = asset_root
    armature.matrix_world = original_world
    asset_root.rotation_euler.z = math.radians(float(yaw_deg))
    bpy.context.view_layer.update()
    print(
        f"[direction-candidate] asset_yaw_deg={float(yaw_deg):+.0f} "
        "axis=blender_positive_z automatic_direction_inference=false",
        flush=True,
    )
    return asset_root


def apply_pose_template_yaw(asset_root, yaw_deg):
    """Expose limb depth in a source image without changing the authored pose."""
    asset_root.rotation_euler.z = math.radians(float(yaw_deg))
    bpy.context.view_layer.update()
    print(
        f"[pose-template] pose_template_yaw_deg={float(yaw_deg):+.1f} "
        "purpose=near_far_limb_depth_evidence "
        "automatic_direction_inference=false",
        flush=True,
    )


def main():
    args = parse_argv()
    if not 1 <= args.samples <= 256:
        raise SystemExit("--samples must be in [1, 256]")
    if not 0.75 <= args.camera_distance_multiplier <= 4.0:
        raise SystemExit("--camera-distance-multiplier must be in [0.75, 4.0]")
    if args.camera_reference_diagonal < 0.0:
        raise SystemExit("--camera-reference-diagonal must be non-negative")
    if args.camera_reference_diagonal and not args.orthographic:
        raise SystemExit("--camera-reference-diagonal requires --orthographic")
    if not 0.0 <= args.quadruped_far_limb_offset_ratio <= (
        MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO
    ):
        raise SystemExit(
            "--quadruped-far-limb-offset-ratio must be in "
            f"[0, {MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO}]"
        )
    if not 0.0 <= args.quadruped_far_limb_action_pose_ratio <= (
        MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO
    ):
        raise SystemExit(
            "--quadruped-far-limb-action-pose-ratio must be in "
            f"[0, {MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO}]"
        )
    if (
        args.quadruped_far_limb_offset_ratio
        and args.quadruped_far_limb_action_pose_ratio
    ):
        raise SystemExit("choose only one quadruped far-limb pose mode")
    if not 9 <= args.quadruped_pose_samples <= 241:
        raise SystemExit("--quadruped-pose-samples must be in [9, 241]")
    if args.quadruped_far_limb_offset_ratio and not args.rest_pose:
        raise SystemExit("--quadruped-far-limb-offset-ratio requires --rest-pose")
    if args.quadruped_far_limb_action_pose_ratio and not args.rest_pose:
        raise SystemExit(
            "--quadruped-far-limb-action-pose-ratio requires --rest-pose"
        )
    if args.pose_template_clay_color and not args.rest_pose:
        raise SystemExit("--pose-template-clay-color requires --rest-pose")
    if not -30.0 <= args.pose_template_yaw_deg <= 30.0:
        raise SystemExit("--pose-template-yaw-deg must be in [-30, 30]")
    if args.pose_template_yaw_deg and not args.rest_pose:
        raise SystemExit("--pose-template-yaw-deg requires --rest-pose")
    if args.pose_template_yaw_deg and not args.orthographic:
        raise SystemExit("--pose-template-yaw-deg requires --orthographic")
    if args.pose_template_yaw_deg and args.asset_yaw_deg:
        raise SystemExit("--pose-template-yaw-deg cannot be combined with --asset-yaw-deg")
    if args.pose_template_clay_color and args.review_clay_color:
        raise SystemExit("choose only one clay material override")
    allowed_yaws = (-90.0, 0.0, 90.0, 180.0)
    if not any(math.isclose(args.asset_yaw_deg, value, abs_tol=1e-9) for value in allowed_yaws):
        raise SystemExit("asset_yaw_deg must be one of -90, 0, 90, 180")
    if not 0.0 <= args.trajectory_distance_ratio <= 2.0:
        raise SystemExit("--trajectory-distance-ratio must be in [0, 2]")
    if args.rest_pose and args.trajectory_distance_ratio:
        raise SystemExit("--trajectory-distance-ratio is only valid for animation actions")
    os.makedirs(args.output_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=args.input)
    else:
        raise SystemExit(f"unsupported input {ext}")

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not meshes or not armatures:
        raise SystemExit("input must contain a mesh and armature")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    for mesh in meshes:
        if mesh != body:
            mesh.hide_viewport = True
            mesh.hide_render = True
    armature = armatures[0]
    armature_modifiers = [
        modifier
        for mesh in meshes
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    if args.preserve_volume:
        if not armature_modifiers:
            raise SystemExit("--preserve-volume requires a linked armature modifier")
        for modifier in armature_modifiers:
            modifier.use_deform_preserve_volume = True
        print(
            "[anim] preserve_volume=true purpose=render_only_diagnostic "
            "input_asset_unchanged=true",
            flush=True,
        )
    asset_root = apply_asset_cardinal_yaw(armature, args.asset_yaw_deg)
    if args.pose_template_yaw_deg:
        apply_pose_template_yaw(asset_root, args.pose_template_yaw_deg)
    if args.pose_template_clay_color:
        apply_pose_template_clay_material(body, args.pose_template_clay_color)
    if args.review_clay_color:
        apply_pose_template_clay_material(body, args.review_clay_color)
    if args.rest_pose:
        armature.animation_data_clear()
        if (
            args.quadruped_far_limb_offset_ratio
            or args.quadruped_far_limb_action_pose_ratio
        ):
            armature.data.pose_position = "POSE"
            reset_pose_basis(armature)
            if args.quadruped_far_limb_action_pose_ratio:
                apply_quadruped_far_limb_action_pose(
                    armature,
                    body,
                    args.quadruped_far_limb_action_pose_ratio,
                    args.quadruped_pose_action,
                    args.quadruped_pose_samples,
                )
            else:
                apply_quadruped_far_limb_offset(
                    armature, body, args.quadruped_far_limb_offset_ratio
                )
        else:
            armature.data.pose_position = "REST"
        action = None
        action_label = (
            "authored_rest_pose_with_native_action_far_limbs"
            if args.quadruped_far_limb_action_pose_ratio
            else "authored_rest_pose_with_far_limb_offset"
            if args.quadruped_far_limb_offset_ratio
            else "authored_rest_pose"
        )
        start = end = 1.0
    else:
        action = choose_action(args.action)
        armature.animation_data_create()
        armature.animation_data.action = action
        action_label = action.name
        start, end = action.frame_range
        if end <= start:
            end = start + max(1, args.n_frames - 1)
    bpy.context.view_layer.update()
    print(f"[anim] body={body.name} hidden_meshes={[m.name for m in meshes if m != body]} "
          f"armature={armature.name} "
          f"action={action_label} frame_range=({start:.2f}, {end:.2f})", flush=True)

    mn, mx = mesh_bbox([body])
    center = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    diag = math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3)))
    framing_diag = args.camera_reference_diagonal or diag
    radius = framing_diag * args.camera_distance_multiplier
    radius += diag * 0.5 * args.trajectory_distance_ratio
    print(
        f"[anim] bbox_min={mn} bbox_max={mx} center={center} diag={diag:.3f} "
        f"camera_reference_diagonal={framing_diag:.3f}",
        flush=True,
    )
    if args.ground_plane:
        add_review_ground(center, mn[2], diag)

    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (center[0], center[1] - radius, center[2] + radius)
    light.rotation_euler = (math.radians(60), 0, 0)

    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.60, 0.65, 1.0)
    bg.inputs[1].default_value = 0.7

    cam_data = bpy.data.cameras.new(name="Cam")
    if args.orthographic:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = framing_diag * 1.05
    else:
        cam_data.lens = 55
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    if args.view == "front":
        cam.location = (center[0] + radius, center[1], center[2] + diag * 0.12)
    elif args.view == "quarter":
        cam.location = (center[0] + radius * 0.85, center[1] - radius * 0.85, center[2] + diag * 0.18)
    else:
        cam.location = (center[0], center[1] - radius, center[2] + diag * 0.12)
    look_at(cam, center)

    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
    else:
        scene.eevee.taa_render_samples = args.samples

    trajectory_base_location = asset_root.location.copy()
    trajectory_distance = diag * args.trajectory_distance_ratio
    if trajectory_distance:
        print(
            f"[direction-candidate] trajectory_axis=world_positive_x "
            f"distance={trajectory_distance:.6f} start_fraction=-0.5 end_fraction=0.5",
            flush=True,
        )
    for i in range(args.n_frames):
        t = 0.0 if args.n_frames == 1 else i / (args.n_frames - 1)
        frame = start + (end - start) * t
        scene.frame_set(int(round(frame)))
        asset_root.location = trajectory_base_location + mathutils.Vector(
            ((t - 0.5) * trajectory_distance, 0.0, 0.0)
        )
        bpy.context.view_layer.update()
        scene.render.filepath = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[anim] frame {i + 1}/{args.n_frames} source_frame={frame:.2f}", flush=True)

    print(f"RENDER_GLB_ANIM_OK {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
