"""Repair visible native-mesh stretch without changing approved animation.

The generated animal mesh, fitted skeleton, PBR material, and already-approved
Walk/Idle actions are treated as immutable authorities.  Only vertex weights
are changed.  Each pass evaluates the real actions, finds short topology edges
whose posed length grows far beyond their rest length, and equalizes the skin
transform on both endpoints.  Re-evaluating after every pass moves the repair
frontier outward when a former seam merely migrates to a neighbouring edge.

This is deliberately motion-aware.  Static weight discontinuity alone is not
an adequate proxy for a visible membrane or hole: a large discontinuity can be
harmless when two bones move together, while a smaller discontinuity can tear
open when their relative motion is large.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.blender_audit_skinned_deformation import (  # noqa: E402
    edge_lengths,
    evaluated_geometry,
)
from tools.blender_diagnose_skinned_deformation import (  # noqa: E402
    bone_records,
)
from tools.blender_smooth_generated_quadruped_joint_weights import (  # noqa: E402
    extract_weights,
    install_weights,
    linked_armatures,
    lowest_common_ancestor_name,
    real_meshes,
    top_k_normalize,
)
from tools.generated_quadruped_semantics import (  # noqa: E402
    infer_quadruped_semantics,
)


SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument("--walking-samples", type=int, default=21)
    parser.add_argument("--idle-samples", type=int, default=9)
    parser.add_argument("--maximum-passes", type=int, default=6)
    parser.add_argument("--inner-iterations", type=int, default=4)
    parser.add_argument(
        "--extension-threshold",
        type=float,
        default=0.006,
        help="Maximum positive edge extension divided by rest bbox diagonal.",
    )
    parser.add_argument("--minimum-stretch-ratio", type=float, default=1.8)
    parser.add_argument(
        "--maximum-rest-edge-ratio",
        type=float,
        default=0.04,
        help="Ignore intentionally long topology edges above this bbox ratio.",
    )
    parser.add_argument("--blend", type=float, default=0.9)
    parser.add_argument(
        "--component-rings",
        type=int,
        default=0,
        help=(
            "Graph-ring falloff used only by component-parent-lock. A nonzero "
            "value prevents a rigid repaired patch from creating a new seam."
        ),
    )
    parser.add_argument(
        "--repair-mode",
        choices=("edge-average", "component-lock", "component-parent-lock"),
        default="edge-average",
        help=(
            "edge-average is the conservative first pass. component-lock "
            "assigns one shared transform mixture to every connected residual "
            "stretch patch. component-parent-lock binds each residual patch "
            "toward the lowest common ancestor of its dominant bones; this is "
            "intended for generated belly/limb scraps incorrectly pulled by a "
            "child joint."
        ),
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--minimum-weight", type=float, default=1.0e-5)
    parser.add_argument("--maximum-seed-edges", type=int, default=4096)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe input: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def action_fingerprint(action) -> str:
    records = []
    for curve in sorted(
        action.fcurves,
        key=lambda item: (item.data_path, item.array_index),
    ):
        records.append(
            {
                "data_path": curve.data_path,
                "array_index": int(curve.array_index),
                "keyframes": [
                    [
                        float(point.co.x),
                        float(point.co.y),
                        point.interpolation,
                    ]
                    for point in curve.keyframe_points
                ],
            }
        )
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def action_curve_stats(action):
    return {
        "fcurves": len(action.fcurves),
        "keyframes": sum(len(curve.keyframe_points) for curve in action.fcurves),
        "frame_range": [float(value) for value in action.frame_range],
    }


def topology_fingerprint(edges, faces) -> str:
    digest = hashlib.sha256()
    for value in (edges, faces):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def find_actions():
    result = {}
    for canonical, hint in (("Walking", "walk"), ("Idle", "idle")):
        candidates = [item for item in bpy.data.actions if hint in item.name.lower()]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one {canonical} action, got "
                f"{[item.name for item in candidates]}"
            )
        result[canonical] = candidates[0]
    return result


def semantic_labels(armature, rest_vertices, front_axis):
    semantics = infer_quadruped_semantics(
        bone_records(armature),
        bbox_min=rest_vertices.min(axis=0),
        bbox_extent=np.ptp(rest_vertices, axis=0),
        front_axis=front_axis,
    )
    labels = {}
    for label, chain in semantics.chains().items():
        for name in chain:
            if name in labels:
                raise RuntimeError(f"duplicate semantic bone: {name}")
            labels[name] = label
    if set(labels) != set(armature.data.bones.keys()):
        raise RuntimeError("semantic decomposition does not cover the skeleton")
    return semantics, labels


def evaluate_edge_error(
    *,
    scene,
    depsgraph,
    armature,
    mesh,
    actions,
    sample_counts,
    rest_vertices,
    edges,
    faces,
    rest_lengths,
    rest_diagonal,
):
    maximum_extension = np.zeros(len(edges), dtype=np.float64)
    maximum_stretch = np.ones(len(edges), dtype=np.float64)
    worst_frame = np.full(len(edges), -1, dtype=np.int64)
    valid = rest_lengths > rest_diagonal * 1.0e-6
    action_records = []
    armature.data.pose_position = "POSE"
    armature.animation_data_create()
    for canonical, action in actions.items():
        armature.animation_data.action = action
        start, end = action.frame_range
        sampled_frames = sorted(
            {
                int(round(float(value)))
                for value in np.linspace(start, end, sample_counts[canonical])
            }
        )
        action_maximum = 0.0
        for frame in sampled_frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            posed, current_edges, current_faces = evaluated_geometry(mesh, depsgraph)
            if not np.array_equal(current_edges, edges) or not np.array_equal(
                current_faces, faces
            ):
                raise RuntimeError("evaluated topology changed during animation")
            posed_lengths = edge_lengths(posed, edges)
            extension = np.zeros(len(edges), dtype=np.float64)
            stretch = np.ones(len(edges), dtype=np.float64)
            extension[valid] = np.maximum(
                posed_lengths[valid] - rest_lengths[valid], 0.0
            ) / rest_diagonal
            stretch[valid] = np.maximum(
                posed_lengths[valid] / rest_lengths[valid],
                rest_lengths[valid] / np.maximum(posed_lengths[valid], 1.0e-12),
            )
            improved = extension > maximum_extension
            maximum_extension[improved] = extension[improved]
            worst_frame[improved] = frame
            maximum_stretch = np.maximum(maximum_stretch, stretch)
            action_maximum = max(action_maximum, float(extension.max(initial=0.0)))
        action_records.append(
            {
                "action": canonical,
                "resolved_action": action.name,
                "sampled_frames": sampled_frames,
                "maximum_extension_ratio_of_rest_diagonal": action_maximum,
            }
        )
    return maximum_extension, maximum_stretch, worst_frame, action_records


def select_seed_edges(
    *,
    extension,
    stretch,
    rest_lengths,
    rest_diagonal,
    extension_threshold,
    minimum_stretch_ratio,
    maximum_rest_edge_ratio,
    maximum_seed_edges,
):
    mask = (
        (extension > extension_threshold)
        & (stretch > minimum_stretch_ratio)
        & (rest_lengths / rest_diagonal < maximum_rest_edge_ratio)
    )
    indices = np.flatnonzero(mask)
    if len(indices) > maximum_seed_edges:
        order = np.argpartition(extension[indices], -maximum_seed_edges)[
            -maximum_seed_edges:
        ]
        indices = indices[order]
    return indices[np.argsort(extension[indices])[::-1]]


def seed_edge_components(edges, seed_edges):
    vertices = np.unique(edges[seed_edges].reshape(-1)).astype(np.int64)
    parent = {int(vertex): int(vertex) for vertex in vertices}

    def find(value):
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for edge_index in seed_edges:
        union(*edges[edge_index])
    groups = {}
    for vertex in vertices:
        groups.setdefault(find(vertex), []).append(int(vertex))
    return [np.asarray(value, dtype=np.int64) for value in groups.values()]


def repair_seed_edges(
    weights,
    edges,
    seed_edges,
    severity,
    iterations,
    blend,
    mode,
    bone_names,
    armature,
    labels,
    component_rings,
):
    result = weights.copy()
    first = edges[seed_edges, 0].astype(np.int64)
    second = edges[seed_edges, 1].astype(np.int64)
    components = seed_edge_components(edges, seed_edges)
    mode_details = {}
    if mode == "component-lock":
        for component in components:
            target = result[component].mean(axis=0)
            result[component] = (
                (1.0 - blend) * result[component] + blend * target[None, :]
            )
            result[component] /= result[component].sum(axis=1)[:, None]
        return (
            result,
            np.unique(np.concatenate(components)),
            components,
            mode_details,
        )
    if mode == "component-parent-lock":
        bone_index = {name: index for index, name in enumerate(bone_names)}
        adjacency = [[] for _ in range(len(result))]
        for first_vertex, second_vertex in edges:
            first_vertex = int(first_vertex)
            second_vertex = int(second_vertex)
            adjacency[first_vertex].append(second_vertex)
            adjacency[second_vertex].append(first_vertex)
        ancestor_counts = {}
        proposals = {}
        for component in components:
            dominant = np.argmax(result[component], axis=1)
            ancestor = lowest_common_ancestor_name(
                armature,
                (bone_names[int(index)] for index in dominant),
            )
            for seed in component:
                seed = int(seed)
                dominant_bone = bone_names[int(np.argmax(result[seed]))]
                seed_chain = labels[dominant_bone]
                queue = [(seed, 0)]
                visited = {seed}
                while queue:
                    vertex, distance = queue.pop(0)
                    proposals.setdefault(vertex, []).append((ancestor, distance))
                    if distance >= component_rings:
                        continue
                    for neighbour in adjacency[vertex]:
                        if neighbour in visited:
                            continue
                        neighbour_bone = bone_names[int(np.argmax(result[neighbour]))]
                        if labels[neighbour_bone] != seed_chain:
                            continue
                        visited.add(neighbour)
                        queue.append((neighbour, distance + 1))
        for vertex, values in proposals.items():
            minimum_distance = min(distance for _ancestor, distance in values)
            ancestor = lowest_common_ancestor_name(
                armature,
                (candidate for candidate, _distance in values),
            )
            falloff = (component_rings + 1 - minimum_distance) / float(
                component_rings + 1
            )
            alpha = min(1.0, blend * falloff)
            target = np.zeros(result.shape[1], dtype=np.float64)
            target[bone_index[ancestor]] = 1.0
            result[vertex] = (1.0 - alpha) * result[vertex] + alpha * target
            result[vertex] /= result[vertex].sum()
            ancestor_counts[ancestor] = ancestor_counts.get(ancestor, 0) + 1
        mode_details["ancestor_vertex_counts"] = ancestor_counts
        mode_details["hierarchical_parent_lock"] = True
        mode_details["component_rings"] = int(component_rings)
        return (
            result,
            np.asarray(sorted(proposals), dtype=np.int64),
            components,
            mode_details,
        )
    if mode != "edge-average":
        raise RuntimeError(f"unsupported repair mode: {mode}")
    edge_mass = 0.25 + 0.75 * severity
    touched = np.unique(np.concatenate((first, second)))
    for _iteration in range(iterations):
        shared = 0.5 * (result[first] + result[second])
        accumulated = np.zeros_like(result)
        mass = np.zeros(len(result), dtype=np.float64)
        weighted_shared = shared * edge_mass[:, None]
        np.add.at(accumulated, first, weighted_shared)
        np.add.at(accumulated, second, weighted_shared)
        np.add.at(mass, first, edge_mass)
        np.add.at(mass, second, edge_mass)
        target = accumulated[touched] / mass[touched, None]
        local_alpha = blend * np.minimum(1.0, mass[touched])
        result[touched] = (
            (1.0 - local_alpha[:, None]) * result[touched]
            + local_alpha[:, None] * target
        )
        result[touched] /= result[touched].sum(axis=1)[:, None]
    return result, touched, components, mode_details


def dominant_semantic_counts(weights, bone_names, labels, vertices):
    dominant = np.argmax(weights[vertices], axis=1)
    counts = {}
    for index in dominant:
        label = labels[bone_names[int(index)]]
        counts[label] = counts.get(label, 0) + 1
    return counts


def add_nla_tracks(armature, actions):
    armature.animation_data_create()
    armature.animation_data.action = None
    while armature.animation_data.nla_tracks:
        armature.animation_data.nla_tracks.remove(armature.animation_data.nla_tracks[0])
    for canonical, action in actions.items():
        start, end = [int(round(value)) for value in action.frame_range]
        track = armature.animation_data.nla_tracks.new()
        track.name = canonical
        strip = track.strips.new(canonical, start, action)
        strip.name = canonical
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_animated(mesh, armature, actions, output):
    armature.data.pose_position = "POSE"
    add_nla_tracks(armature, actions)
    bpy.context.scene.frame_set(
        int(round(min(action.frame_range[0] for action in actions.values())))
    )
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    armature.select_set(True)
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
        export_materials="EXPORT",
    )


def main():
    args = parse_argv()
    source = require_input(args.input)
    output = require_output(args.output, "output GLB")
    manifest = require_output(args.manifest, "manifest")
    if not 4 <= args.walking_samples <= 81 or not 4 <= args.idle_samples <= 81:
        raise SystemExit("sample counts must be in [4, 81]")
    if not 1 <= args.maximum_passes <= 12:
        raise SystemExit("--maximum-passes must be in [1, 12]")
    if not 1 <= args.inner_iterations <= 20:
        raise SystemExit("--inner-iterations must be in [1, 20]")
    if not 0.0 < args.extension_threshold <= 0.05:
        raise SystemExit("--extension-threshold must be in (0, 0.05]")
    if not 1.0 < args.minimum_stretch_ratio <= 20.0:
        raise SystemExit("--minimum-stretch-ratio must be in (1, 20]")
    if not 0.0 < args.maximum_rest_edge_ratio <= 0.1:
        raise SystemExit("--maximum-rest-edge-ratio must be in (0, 0.1]")
    if not 0.0 < args.blend <= 1.0:
        raise SystemExit("--blend must be in (0, 1]")
    if not 0 <= args.component_rings <= 12:
        raise SystemExit("--component-rings must be in [0, 12]")
    if not 1 <= args.top_k <= 8:
        raise SystemExit("--top-k must be in [1, 8]")
    if not 1 <= args.maximum_seed_edges <= 65536:
        raise SystemExit("--maximum-seed-edges must be in [1, 65536]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = real_meshes()
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise RuntimeError("input needs exactly one exported mesh and one armature")
    mesh = meshes[0]
    armature = armatures[0]
    if linked_armatures(mesh) != {armature}:
        raise RuntimeError("mesh is not linked to the only armature")
    actions = find_actions()
    action_hashes_before = {
        canonical: action_fingerprint(action) for canonical, action in actions.items()
    }
    action_stats_before = {
        canonical: action_curve_stats(action) for canonical, action in actions.items()
    }

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    armature.data.pose_position = "REST"
    armature.animation_data_create()
    armature.animation_data.action = None
    scene.frame_set(0)
    bpy.context.view_layer.update()
    rest_vertices, edges, faces = evaluated_geometry(mesh, depsgraph)
    rest_diagonal = float(np.linalg.norm(np.ptp(rest_vertices, axis=0)))
    rest_lengths = edge_lengths(rest_vertices, edges)
    topology_before = topology_fingerprint(edges, faces)
    semantics, labels = semantic_labels(armature, rest_vertices, args.front_axis)
    bone_names = [bone.name for bone in armature.data.bones]
    weights_before = extract_weights(mesh, bone_names)
    weights = weights_before.copy()
    sample_counts = {
        "Walking": args.walking_samples,
        "Idle": args.idle_samples,
    }

    passes = []
    all_changed = set()
    for pass_index in range(args.maximum_passes):
        extension, stretch, worst_frame, action_records = evaluate_edge_error(
            scene=scene,
            depsgraph=depsgraph,
            armature=armature,
            mesh=mesh,
            actions=actions,
            sample_counts=sample_counts,
            rest_vertices=rest_vertices,
            edges=edges,
            faces=faces,
            rest_lengths=rest_lengths,
            rest_diagonal=rest_diagonal,
        )
        seeds = select_seed_edges(
            extension=extension,
            stretch=stretch,
            rest_lengths=rest_lengths,
            rest_diagonal=rest_diagonal,
            extension_threshold=args.extension_threshold,
            minimum_stretch_ratio=args.minimum_stretch_ratio,
            maximum_rest_edge_ratio=args.maximum_rest_edge_ratio,
            maximum_seed_edges=args.maximum_seed_edges,
        )
        record = {
            "pass": pass_index,
            "action_measurements": action_records,
            "maximum_extension_ratio_of_rest_diagonal": float(
                extension.max(initial=0.0)
            ),
            "seed_edge_count": int(len(seeds)),
            "maximum_seed_extension_ratio_of_rest_diagonal": (
                float(extension[seeds[0]]) if len(seeds) else 0.0
            ),
            "seed_edge_sample": [
                {
                    "edge_index": int(index),
                    "vertices": [int(value) for value in edges[index]],
                    "extension_ratio_of_rest_diagonal": float(extension[index]),
                    "stretch_ratio": float(stretch[index]),
                    "worst_frame": int(worst_frame[index]),
                }
                for index in seeds[:64]
            ],
        }
        print(
            "[weight-repair] "
            f"pass={pass_index + 1}/{args.maximum_passes} "
            f"max_extension={record['maximum_extension_ratio_of_rest_diagonal']:.6f} "
            f"seed_edges={len(seeds)}",
            flush=True,
        )
        if not len(seeds):
            record["changed_vertex_count"] = 0
            passes.append(record)
            break
        seed_extension = extension[seeds]
        span = max(float(seed_extension.max() - args.extension_threshold), 1.0e-12)
        severity = np.clip(
            (seed_extension - args.extension_threshold) / span,
            0.0,
            1.0,
        )
        repaired, touched, components, mode_details = repair_seed_edges(
            weights,
            edges,
            seeds,
            severity,
            args.inner_iterations,
            args.blend,
            args.repair_mode,
            bone_names,
            armature,
            labels,
            args.component_rings,
        )
        repaired = top_k_normalize(repaired, args.top_k, args.minimum_weight)
        record["changed_vertex_count"] = int(len(touched))
        record["seed_component_count"] = len(components)
        record["seed_component_vertex_counts_descending"] = sorted(
            (len(component) for component in components), reverse=True
        )[:64]
        record["repair_mode_details"] = mode_details
        record["touched_dominant_semantics"] = dominant_semantic_counts(
            weights, bone_names, labels, touched
        )
        record["mean_touched_l1_weight_delta"] = float(
            np.abs(repaired[touched] - weights[touched]).sum(axis=1).mean()
        )
        passes.append(record)
        all_changed.update(int(value) for value in touched)
        weights = repaired
        install_weights(mesh, weights, bone_names, args.minimum_weight)
        bpy.context.view_layer.update()
        print(
            f"[weight-repair] pass={pass_index + 1} changed_vertices={len(touched)}",
            flush=True,
        )

    final_extension, final_stretch, _final_frame, final_actions = evaluate_edge_error(
        scene=scene,
        depsgraph=depsgraph,
        armature=armature,
        mesh=mesh,
        actions=actions,
        sample_counts=sample_counts,
        rest_vertices=rest_vertices,
        edges=edges,
        faces=faces,
        rest_lengths=rest_lengths,
        rest_diagonal=rest_diagonal,
    )
    final_seeds = select_seed_edges(
        extension=final_extension,
        stretch=final_stretch,
        rest_lengths=rest_lengths,
        rest_diagonal=rest_diagonal,
        extension_threshold=args.extension_threshold,
        minimum_stretch_ratio=args.minimum_stretch_ratio,
        maximum_rest_edge_ratio=args.maximum_rest_edge_ratio,
        maximum_seed_edges=args.maximum_seed_edges,
    )
    armature.data.pose_position = "REST"
    armature.animation_data.action = None
    scene.frame_set(0)
    bpy.context.view_layer.update()
    final_vertices, final_edges, final_faces = evaluated_geometry(mesh, depsgraph)
    topology_after = topology_fingerprint(final_edges, final_faces)
    if topology_after != topology_before:
        raise RuntimeError("weight repair changed rest topology")
    if final_vertices.shape != rest_vertices.shape:
        raise RuntimeError("weight repair changed rest vertex count")
    maximum_rest_geometry_delta = float(
        np.linalg.norm(final_vertices - rest_vertices, axis=1).max(initial=0.0)
    )
    if maximum_rest_geometry_delta > max(rest_diagonal * 1.0e-6, 1.0e-8):
        raise RuntimeError(
            "weight repair changed rest geometry: "
            f"maximum_delta={maximum_rest_geometry_delta}"
        )
    action_hashes_after = {
        canonical: action_fingerprint(action) for canonical, action in actions.items()
    }
    if action_hashes_after != action_hashes_before:
        raise RuntimeError("weight repair changed approved action curves")
    action_stats_after = {
        canonical: action_curve_stats(action) for canonical, action in actions.items()
    }
    if action_stats_after != action_stats_before:
        raise RuntimeError("weight repair changed approved action curve counts")

    export_animated(mesh, armature, actions, output)
    delta = np.abs(weights - weights_before).sum(axis=1)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "authority_contract": {
            "native_mesh_geometry_preserved": True,
            "native_mesh_topology_preserved": True,
            "pbr_material_preserved": True,
            "fitted_skeleton_rest_matrices_preserved": True,
            "approved_animation_curves_preserved": True,
            "only_vertex_weights_modified_in_memory": True,
            "source_animation_fingerprints": action_hashes_before,
            "post_repair_animation_fingerprints": action_hashes_after,
            "source_animation_curve_stats": action_stats_before,
            "post_repair_animation_curve_stats": action_stats_after,
            "rest_geometry_topology_fingerprint_before": topology_before,
            "rest_geometry_topology_fingerprint_after": topology_after,
            "maximum_rest_geometry_delta": maximum_rest_geometry_delta,
        },
        "front_axis": args.front_axis,
        "semantic_rig": {
            "root": semantics.root,
            "chains": {
                name: list(value) for name, value in semantics.chains().items()
            },
        },
        "parameters": {
            "walking_samples": args.walking_samples,
            "idle_samples": args.idle_samples,
            "maximum_passes": args.maximum_passes,
            "inner_iterations": args.inner_iterations,
            "extension_threshold": args.extension_threshold,
            "minimum_stretch_ratio": args.minimum_stretch_ratio,
            "maximum_rest_edge_ratio": args.maximum_rest_edge_ratio,
            "blend": args.blend,
            "component_rings": args.component_rings,
            "repair_mode": args.repair_mode,
            "top_k": args.top_k,
            "minimum_weight": args.minimum_weight,
            "maximum_seed_edges": args.maximum_seed_edges,
        },
        "passes": passes,
        "weight_change": {
            "changed_vertices": int(np.count_nonzero(delta > 1.0e-8)),
            "touched_vertices_across_passes": len(all_changed),
            "mean_l1_per_vertex": float(delta.mean()),
            "maximum_l1_per_vertex": float(delta.max(initial=0.0)),
        },
        "final_measurements": {
            "actions": final_actions,
            "maximum_extension_ratio_of_rest_diagonal": float(
                final_extension.max(initial=0.0)
            ),
            "remaining_seed_edge_count": int(len(final_seeds)),
            "maximum_remaining_seed_extension_ratio_of_rest_diagonal": (
                float(final_extension[final_seeds[0]]) if len(final_seeds) else 0.0
            ),
        },
        "status": (
            "research_candidate_pending_readback_and_visual_qa"
            if not len(final_seeds)
            else "research_candidate_repair_incomplete"
        ),
        "formal_dataset_registration_authorized": False,
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "MOTION_AWARE_QUADRUPED_WEIGHT_REPAIR_OK "
        f"max_extension={payload['final_measurements']['maximum_extension_ratio_of_rest_diagonal']:.6f} "
        f"remaining_seeds={len(final_seeds)} changed={payload['weight_change']['changed_vertices']} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
