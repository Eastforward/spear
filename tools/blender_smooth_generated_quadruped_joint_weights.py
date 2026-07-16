"""Repair hard skin-weight seams on a generated native quadruped mesh.

SkinTokens can assign geometrically adjacent vertices almost one-hot weights
for consecutive bones on the same leg.  A normal Walk then stretches the tiny
rest edge into a visible spike even though the skeleton and global direction
are correct.  This tool keeps the native mesh, UVs, PBR material, and fitted
skeleton, but:

1. removes weight leakage from a limb vertex to a different locomotion limb;
2. binds fused cross-limb bridges to their shared torso ancestor;
3. stabilizes hard torso-to-limb attachment boundaries around the hip/shoulder;
4. smooths only bounded graph neighbourhoods around same-chain seams; and
5. restores normalized top-k weights before exporting a new immutable GLB.

It does not add animation.  The repaired static GLB must be retargeted and pass
the independent deformation, contact, and visual gates afterwards.
"""

from __future__ import annotations

import argparse
import bmesh
from collections import deque
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

from tools.blender_diagnose_skinned_deformation import (  # noqa: E402
    LIMB_CHAINS,
    bone_records,
)
from tools.generated_quadruped_semantics import (  # noqa: E402
    infer_quadruped_semantics,
    quadruped_semantic_labels,
)


SCHEMA = "avengine_generated_quadruped_joint_weight_smoothing_v3"


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
    parser.add_argument("--seam-l1-threshold", type=float, default=0.4)
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--minimum-weight", type=float, default=1.0e-5)
    parser.add_argument("--bridge-rings", type=int, default=2)
    parser.add_argument("--bridge-blend", type=float, default=1.0)
    parser.add_argument("--attachment-l1-threshold", type=float, default=0.75)
    parser.add_argument("--attachment-rings", type=int, default=3)
    parser.add_argument("--attachment-blend", type=float, default=1.0)
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


def real_meshes():
    hidden = bpy.data.collections.get("glTF_not_exported")
    hidden_objects = set(hidden.objects) if hidden is not None else set()
    return [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and item not in hidden_objects
    ]


def linked_armatures(mesh):
    result = set()
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        result.add(mesh.parent)
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            result.add(modifier.object)
    return result


def world_vertices(mesh):
    matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
    local = np.asarray([vertex.co[:] for vertex in mesh.data.vertices], dtype=np.float64)
    homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float64)))
    return (homogeneous @ matrix.T)[:, :3]


def infer_labels(armature, mesh, front_axis):
    vertices = world_vertices(mesh)
    records = bone_records(armature)
    semantics = infer_quadruped_semantics(
        records,
        bbox_min=vertices.min(axis=0),
        bbox_extent=np.ptp(vertices, axis=0),
        front_axis=front_axis,
    )
    labels = quadruped_semantic_labels(
        semantics,
        records,
        bbox_min=vertices.min(axis=0),
        bbox_extent=np.ptp(vertices, axis=0),
        front_axis=front_axis,
    )
    if set(labels) != set(armature.data.bones.keys()):
        raise RuntimeError("semantic decomposition does not cover the skeleton")
    return semantics, labels


def extract_weights(mesh, bone_names):
    group_by_index = {group.index: group.name for group in mesh.vertex_groups}
    bone_index = {name: index for index, name in enumerate(bone_names)}
    weights = np.zeros((len(mesh.data.vertices), len(bone_names)), dtype=np.float64)
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            name = group_by_index.get(membership.group)
            if name in bone_index:
                weights[vertex.index, bone_index[name]] = float(membership.weight)
    sums = weights.sum(axis=1)
    if np.any(sums <= 0.0):
        raise RuntimeError("input contains unweighted vertices")
    weights /= sums[:, None]
    return weights


def mesh_edges_and_adjacency(mesh):
    edges = np.asarray([edge.vertices[:] for edge in mesh.data.edges], dtype=np.int64)
    adjacency = [[] for _ in mesh.data.vertices]
    for first, second in edges:
        first = int(first)
        second = int(second)
        adjacency[first].append(second)
        adjacency[second].append(first)
    return edges, adjacency


def dominant_semantics(weights, bone_names, labels):
    dominant_bones = np.argmax(weights, axis=1)
    chains = np.asarray(
        [labels[bone_names[index]] for index in dominant_bones], dtype=object
    )
    return dominant_bones, chains


def cross_limb_weight_count(weights, dominant_chains, bone_names, labels, epsilon):
    count = 0
    mass = 0.0
    for vertex_index, chain in enumerate(dominant_chains):
        if chain not in LIMB_CHAINS:
            continue
        for bone_index, name in enumerate(bone_names):
            if labels[name] in LIMB_CHAINS and labels[name] != chain:
                value = float(weights[vertex_index, bone_index])
                if value > epsilon:
                    count += 1
                    mass += value
    return count, mass


def remove_cross_limb_leakage(weights, dominant_chains, bone_names, labels):
    result = weights.copy()
    removed_mass = 0.0
    removed_entries = 0
    for vertex_index, chain in enumerate(dominant_chains):
        if chain not in LIMB_CHAINS:
            continue
        for bone_index, name in enumerate(bone_names):
            other = labels[name]
            if other in LIMB_CHAINS and other != chain:
                value = float(result[vertex_index, bone_index])
                if value > 0.0:
                    removed_mass += value
                    removed_entries += 1
                    result[vertex_index, bone_index] = 0.0
    sums = result.sum(axis=1)
    if np.any(sums <= 0.0):
        raise RuntimeError("cross-limb cleanup created an unweighted vertex")
    result /= sums[:, None]
    return result, removed_entries, removed_mass


def lowest_common_ancestor_name(armature, names):
    names = list(names)
    if not names:
        raise RuntimeError("cannot find a common ancestor of no bones")
    ancestor_paths = []
    for name in names:
        bone = armature.data.bones.get(name)
        if bone is None:
            raise RuntimeError(f"unknown bone in common-ancestor query: {name}")
        path = []
        while bone is not None:
            path.append(bone.name)
            bone = bone.parent
        ancestor_paths.append(path)
    first_path = ancestor_paths[0]
    for candidate in first_path:
        if all(candidate in path for path in ancestor_paths[1:]):
            return candidate
    raise RuntimeError(f"bones have no common ancestor: {names}")


def cross_limb_dominant_edges(
    edges,
    dominant_bones,
    dominant_chains,
    bone_names,
    armature,
):
    records = []
    for edge_index, (first, second) in enumerate(edges):
        first = int(first)
        second = int(second)
        first_chain = dominant_chains[first]
        second_chain = dominant_chains[second]
        if (
            first_chain not in LIMB_CHAINS
            or second_chain not in LIMB_CHAINS
            or first_chain == second_chain
        ):
            continue
        first_bone = bone_names[int(dominant_bones[first])]
        second_bone = bone_names[int(dominant_bones[second])]
        records.append(
            {
                "edge_index": int(edge_index),
                "vertices": [first, second],
                "dominant_bones": [first_bone, second_bone],
                "semantic_chains": [str(first_chain), str(second_chain)],
                "common_ancestor_bone": lowest_common_ancestor_name(
                    armature,
                    (first_bone, second_bone),
                ),
            }
        )
    return records


def stabilize_cross_limb_bridges(
    weights,
    adjacency,
    dominant_chains,
    bridge_edges,
    bone_names,
    armature,
    rings,
    blend,
):
    """Bind fused left/right bridge patches to their shared torso ancestor."""

    proposals = {}
    for record in bridge_edges:
        common_ancestor = record["common_ancestor_bone"]
        for seed in record["vertices"]:
            seed_chain = dominant_chains[seed]
            queue = deque([(seed, 0)])
            visited = {seed}
            while queue:
                vertex, distance = queue.popleft()
                proposals.setdefault(vertex, []).append((common_ancestor, distance))
                if distance >= rings:
                    continue
                for neighbour in adjacency[vertex]:
                    if neighbour in visited or dominant_chains[neighbour] != seed_chain:
                        continue
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))

    bone_index = {name: index for index, name in enumerate(bone_names)}
    result = weights.copy()
    ancestor_counts = {}
    for vertex, values in proposals.items():
        minimum_distance = min(distance for _ancestor, distance in values)
        ancestor = lowest_common_ancestor_name(
            armature,
            (value[0] for value in values),
        )
        ancestor_counts[ancestor] = ancestor_counts.get(ancestor, 0) + 1
        falloff = (rings + 1 - minimum_distance) / float(rings + 1)
        alpha = min(1.0, blend * falloff)
        target = np.zeros(result.shape[1], dtype=np.float64)
        target[bone_index[ancestor]] = 1.0
        result[vertex] = (1.0 - alpha) * result[vertex] + alpha * target
        result[vertex] /= result[vertex].sum()
    return result, {
        "seed_edges": len(bridge_edges),
        "seed_vertices": len(
            {vertex for record in bridge_edges for vertex in record["vertices"]}
        ),
        "affected_vertices": len(proposals),
        "ancestor_vertex_counts": ancestor_counts,
        "rings": int(rings),
        "blend": float(blend),
    }


def axial_limb_attachment_edges(
    weights,
    edges,
    dominant_bones,
    dominant_chains,
    bone_names,
    armature,
    threshold,
):
    """Find abrupt native-mesh edges between the torso and one limb root.

    These are distinct from left/right bridge errors.  SkinTokens can put two
    adjacent shoulder or hip vertices almost one-hot on the torso ancestor and
    the first limb bone respectively.  A full-amplitude gait then pulls that
    tiny edge into a visible spike even though neither endpoint contains a
    cross-limb influence.
    """

    records = []
    for edge_index, (first, second) in enumerate(edges):
        first = int(first)
        second = int(second)
        first_chain = dominant_chains[first]
        second_chain = dominant_chains[second]
        chains = {first_chain, second_chain}
        limb_chains = chains.intersection(LIMB_CHAINS)
        if "axial" not in chains or len(limb_chains) != 1:
            continue
        l1 = float(np.abs(weights[first] - weights[second]).sum())
        if l1 < threshold:
            continue
        first_bone = bone_names[int(dominant_bones[first])]
        second_bone = bone_names[int(dominant_bones[second])]
        records.append(
            {
                "edge_index": int(edge_index),
                "vertices": [first, second],
                "dominant_bones": [first_bone, second_bone],
                "semantic_chains": [str(first_chain), str(second_chain)],
                "limb_chain": str(next(iter(limb_chains))),
                "weight_l1": l1,
                "common_ancestor_bone": lowest_common_ancestor_name(
                    armature,
                    (first_bone, second_bone),
                ),
            }
        )
    records.sort(key=lambda item: item["weight_l1"], reverse=True)
    return records


def stabilize_axial_limb_attachments(
    weights,
    adjacency,
    dominant_chains,
    attachment_edges,
    bone_names,
    armature,
    rings,
    blend,
):
    """Bind a tiny attachment patch toward its shared torso ancestor.

    Expansion is restricted to the seed endpoint's original semantic chain.
    Consequently an axial patch cannot walk through the mesh into a different
    locomotion limb, and a limb patch cannot leak to its opposite side.  The
    graph-ring falloff keeps the distal limb motion intact.
    """

    proposals = {}
    for record in attachment_edges:
        common_ancestor = record["common_ancestor_bone"]
        limb_chain = record["limb_chain"]
        for seed in record["vertices"]:
            seed_chain = dominant_chains[seed]
            if seed_chain not in {"axial", limb_chain}:
                raise RuntimeError("attachment seed has inconsistent semantic chain")
            queue = deque([(seed, 0)])
            visited = {seed}
            while queue:
                vertex, distance = queue.popleft()
                proposals.setdefault(vertex, []).append((common_ancestor, distance))
                if distance >= rings:
                    continue
                for neighbour in adjacency[vertex]:
                    if neighbour in visited or dominant_chains[neighbour] != seed_chain:
                        continue
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))

    bone_index = {name: index for index, name in enumerate(bone_names)}
    result = weights.copy()
    ancestor_counts = {}
    for vertex, values in proposals.items():
        minimum_distance = min(distance for _ancestor, distance in values)
        ancestor = lowest_common_ancestor_name(
            armature,
            (value[0] for value in values),
        )
        ancestor_counts[ancestor] = ancestor_counts.get(ancestor, 0) + 1
        falloff = (rings + 1 - minimum_distance) / float(rings + 1)
        alpha = min(1.0, blend * falloff)
        target = np.zeros(result.shape[1], dtype=np.float64)
        target[bone_index[ancestor]] = 1.0
        result[vertex] = (1.0 - alpha) * result[vertex] + alpha * target
        result[vertex] /= result[vertex].sum()
    return result, {
        "seed_edges": len(attachment_edges),
        "seed_vertices": len(
            {vertex for record in attachment_edges for vertex in record["vertices"]}
        ),
        "affected_vertices": len(proposals),
        "ancestor_vertex_counts": ancestor_counts,
        "rings": int(rings),
        "blend": float(blend),
    }


def adjacent_bone_pairs(semantics):
    pairs = set()
    for label, chain in semantics.chains().items():
        if label not in LIMB_CHAINS:
            continue
        for first, second in zip(chain, chain[1:]):
            pairs.add(frozenset((first, second)))
    return pairs


def hard_joint_seams(
    weights,
    edges,
    dominant_bones,
    dominant_chains,
    bone_names,
    adjacent_pairs,
    threshold,
):
    records = []
    for edge_index, (first, second) in enumerate(edges):
        first = int(first)
        second = int(second)
        chain = dominant_chains[first]
        if chain not in LIMB_CHAINS or dominant_chains[second] != chain:
            continue
        first_bone = bone_names[int(dominant_bones[first])]
        second_bone = bone_names[int(dominant_bones[second])]
        if first_bone == second_bone:
            continue
        if frozenset((first_bone, second_bone)) not in adjacent_pairs:
            continue
        l1 = float(np.abs(weights[first] - weights[second]).sum())
        if l1 >= threshold:
            records.append(
                {
                    "edge_index": edge_index,
                    "vertices": [first, second],
                    "semantic_chain": str(chain),
                    "dominant_bones": [first_bone, second_bone],
                    "weight_l1": l1,
                }
            )
    records.sort(key=lambda item: item["weight_l1"], reverse=True)
    return records


def affected_neighbourhood(seams, adjacency, dominant_chains, rings):
    distance = {}
    queue = deque()
    for seam in seams:
        for vertex in seam["vertices"]:
            if vertex not in distance:
                distance[vertex] = 0
                queue.append(vertex)
    while queue:
        current = queue.popleft()
        current_distance = distance[current]
        if current_distance >= rings:
            continue
        chain = dominant_chains[current]
        for neighbour in adjacency[current]:
            if dominant_chains[neighbour] != chain or neighbour in distance:
                continue
            distance[neighbour] = current_distance + 1
            queue.append(neighbour)
    return distance


def smooth_neighbourhood(
    weights,
    adjacency,
    dominant_chains,
    affected,
    bone_names,
    labels,
    iterations,
    blend,
):
    result = weights.copy()
    ordered = sorted(affected)
    for _iteration in range(iterations):
        updated = result.copy()
        for vertex in ordered:
            chain = dominant_chains[vertex]
            neighbours = [
                index
                for index in adjacency[vertex]
                if dominant_chains[index] == chain
            ]
            if not neighbours:
                continue
            average = result[neighbours].mean(axis=0)
            value = (1.0 - blend) * result[vertex] + blend * average
            if chain in LIMB_CHAINS:
                for bone_index, name in enumerate(bone_names):
                    other = labels[name]
                    if other in LIMB_CHAINS and other != chain:
                        value[bone_index] = 0.0
            total = float(value.sum())
            if total <= 0.0:
                raise RuntimeError("smoothing created an unweighted vertex")
            updated[vertex] = value / total
        result = updated
    return result


def top_k_normalize(weights, top_k, minimum):
    result = weights.copy()
    result[result < minimum] = 0.0
    if top_k < result.shape[1]:
        keep = np.argpartition(result, -top_k, axis=1)[:, -top_k:]
        mask = np.zeros_like(result, dtype=bool)
        rows = np.arange(len(result))[:, None]
        mask[rows, keep] = True
        result[~mask] = 0.0
    sums = result.sum(axis=1)
    if np.any(sums <= 0.0):
        raise RuntimeError("top-k pruning created an unweighted vertex")
    result /= sums[:, None]
    return result


def install_weights(mesh, weights, bone_names, minimum):
    group_by_name = {group.name: group for group in mesh.vertex_groups}
    missing = [name for name in bone_names if name not in group_by_name]
    if missing:
        raise RuntimeError(f"mesh is missing bone vertex groups: {missing}")
    group_indices = [group_by_name[name].index for name in bone_names]
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    for vertex in bm.verts:
        values = vertex[deform]
        values.clear()
        row = weights[vertex.index]
        for bone_index in np.flatnonzero(row >= minimum):
            values[group_indices[int(bone_index)]] = float(row[bone_index])
    bm.to_mesh(mesh.data)
    bm.free()
    mesh.data.update()


def weight_stats(weights, minimum):
    counts = np.count_nonzero(weights >= minimum, axis=1)
    sums = weights.sum(axis=1)
    return {
        "vertices": len(weights),
        "maximum_influences": int(counts.max(initial=0)),
        "minimum_influences": int(counts.min(initial=0)),
        "minimum_weight_sum": float(sums.min(initial=0.0)),
        "maximum_weight_sum": float(sums.max(initial=0.0)),
    }


def export_static(mesh, armature, output):
    for item in bpy.context.scene.objects:
        item.select_set(False)
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_yup=True,
    )


def main():
    args = parse_argv()
    source = require_input(args.input)
    output = require_output(args.output, "output GLB")
    manifest = require_output(args.manifest, "manifest")
    if not 0.0 < args.seam_l1_threshold <= 2.0:
        raise SystemExit("--seam-l1-threshold must be in (0, 2]")
    if not 1 <= args.rings <= 12 or not 1 <= args.iterations <= 20:
        raise SystemExit("--rings/--iterations are outside supported bounds")
    if not 0.0 < args.blend <= 1.0:
        raise SystemExit("--blend must be in (0, 1]")
    if not 1 <= args.top_k <= 8:
        raise SystemExit("--top-k must be in [1, 8]")
    if not 0 <= args.bridge_rings <= 8:
        raise SystemExit("--bridge-rings must be in [0, 8]")
    if not 0.0 < args.bridge_blend <= 1.0:
        raise SystemExit("--bridge-blend must be in (0, 1]")
    if not 0.0 < args.attachment_l1_threshold <= 2.0:
        raise SystemExit("--attachment-l1-threshold must be in (0, 2]")
    if not 0 <= args.attachment_rings <= 8:
        raise SystemExit("--attachment-rings must be in [0, 8]")
    if not 0.0 < args.attachment_blend <= 1.0:
        raise SystemExit("--attachment-blend must be in (0, 1]")

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
    if bpy.data.actions:
        raise RuntimeError("input must be a static rig without animation")

    semantics, labels = infer_labels(armature, mesh, args.front_axis)
    bone_names = [bone.name for bone in armature.data.bones]
    weights_before = extract_weights(mesh, bone_names)
    edges, adjacency = mesh_edges_and_adjacency(mesh)
    dominant_bones, dominant_chains = dominant_semantics(
        weights_before, bone_names, labels
    )
    cross_before = cross_limb_weight_count(
        weights_before,
        dominant_chains,
        bone_names,
        labels,
        args.minimum_weight,
    )
    cleaned, removed_entries, removed_mass = remove_cross_limb_leakage(
        weights_before, dominant_chains, bone_names, labels
    )
    dominant_bones, dominant_chains = dominant_semantics(cleaned, bone_names, labels)
    bridge_edges_before = cross_limb_dominant_edges(
        edges,
        dominant_bones,
        dominant_chains,
        bone_names,
        armature,
    )
    cleaned, bridge_stabilization = stabilize_cross_limb_bridges(
        cleaned,
        adjacency,
        dominant_chains,
        bridge_edges_before,
        bone_names,
        armature,
        args.bridge_rings,
        args.bridge_blend,
    )
    dominant_bones, dominant_chains = dominant_semantics(cleaned, bone_names, labels)
    bridge_edges_after = cross_limb_dominant_edges(
        edges,
        dominant_bones,
        dominant_chains,
        bone_names,
        armature,
    )
    attachment_edges_before = axial_limb_attachment_edges(
        cleaned,
        edges,
        dominant_bones,
        dominant_chains,
        bone_names,
        armature,
        args.attachment_l1_threshold,
    )
    cleaned, attachment_stabilization = stabilize_axial_limb_attachments(
        cleaned,
        adjacency,
        dominant_chains,
        attachment_edges_before,
        bone_names,
        armature,
        args.attachment_rings,
        args.attachment_blend,
    )
    dominant_bones, dominant_chains = dominant_semantics(cleaned, bone_names, labels)
    attachment_edges_after_stabilization = axial_limb_attachment_edges(
        cleaned,
        edges,
        dominant_bones,
        dominant_chains,
        bone_names,
        armature,
        args.attachment_l1_threshold,
    )
    adjacent_pairs = adjacent_bone_pairs(semantics)
    seams_before = hard_joint_seams(
        cleaned,
        edges,
        dominant_bones,
        dominant_chains,
        bone_names,
        adjacent_pairs,
        args.seam_l1_threshold,
    )
    if not seams_before and not bridge_edges_before and not attachment_edges_before:
        raise RuntimeError("no repairable weight discontinuity found; refusing no-op export")
    affected = affected_neighbourhood(
        seams_before, adjacency, dominant_chains, args.rings
    )
    smoothed = smooth_neighbourhood(
        cleaned,
        adjacency,
        dominant_chains,
        affected,
        bone_names,
        labels,
        args.iterations,
        args.blend,
    )
    final = top_k_normalize(smoothed, args.top_k, args.minimum_weight)
    final_dominant_bones, final_dominant_chains = dominant_semantics(
        final, bone_names, labels
    )
    attachment_edges_after_same_chain_smoothing = axial_limb_attachment_edges(
        final,
        edges,
        final_dominant_bones,
        final_dominant_chains,
        bone_names,
        armature,
        args.attachment_l1_threshold,
    )
    final, post_attachment_stabilization = stabilize_axial_limb_attachments(
        final,
        adjacency,
        final_dominant_chains,
        attachment_edges_after_same_chain_smoothing,
        bone_names,
        armature,
        args.attachment_rings,
        args.attachment_blend,
    )
    final = top_k_normalize(final, args.top_k, args.minimum_weight)
    final_dominant_bones, final_dominant_chains = dominant_semantics(
        final, bone_names, labels
    )
    attachment_edges_after = axial_limb_attachment_edges(
        final,
        edges,
        final_dominant_bones,
        final_dominant_chains,
        bone_names,
        armature,
        args.attachment_l1_threshold,
    )
    seams_after = hard_joint_seams(
        final,
        edges,
        final_dominant_bones,
        final_dominant_chains,
        bone_names,
        adjacent_pairs,
        args.seam_l1_threshold,
    )
    cross_after = cross_limb_weight_count(
        final,
        final_dominant_chains,
        bone_names,
        labels,
        args.minimum_weight,
    )
    delta = np.abs(final - weights_before).sum(axis=1)
    install_weights(mesh, final, bone_names, args.minimum_weight)
    export_static(mesh, armature, output)

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
        "front_axis": args.front_axis,
        "semantic_rig": {
            "root": semantics.root,
            "chains": {
                name: list(value) for name, value in semantics.chains().items()
            },
        },
        "parameters": {
            "profile": "native_quadruped_full_gait_v1",
            "seam_l1_threshold": args.seam_l1_threshold,
            "rings": args.rings,
            "iterations": args.iterations,
            "blend": args.blend,
            "top_k": args.top_k,
            "minimum_weight": args.minimum_weight,
            "bridge_rings": args.bridge_rings,
            "bridge_blend": args.bridge_blend,
            "attachment_l1_threshold": args.attachment_l1_threshold,
            "attachment_rings": args.attachment_rings,
            "attachment_blend": args.attachment_blend,
        },
        "cross_limb_cleanup": {
            "entries_before": cross_before[0],
            "weight_mass_before": cross_before[1],
            "entries_removed": removed_entries,
            "weight_mass_removed": removed_mass,
            "entries_after": cross_after[0],
            "weight_mass_after": cross_after[1],
        },
        "cross_limb_bridge_stabilization": {
            **bridge_stabilization,
            "dominant_edges_before": len(bridge_edges_before),
            "dominant_edges_after": len(bridge_edges_after),
            "sample_before": bridge_edges_before[:64],
            "sample_after": bridge_edges_after[:64],
            "topology_deleted": False,
            "bridge_vertices_bound_toward_common_torso_ancestor": True,
        },
        "axial_limb_attachment_stabilization": {
            **attachment_stabilization,
            "dominant_edges_before": len(attachment_edges_before),
            "dominant_edges_after_stabilization": len(
                attachment_edges_after_stabilization
            ),
            "dominant_edges_after_same_chain_smoothing": len(
                attachment_edges_after_same_chain_smoothing
            ),
            "post_same_chain_smoothing_pass": post_attachment_stabilization,
            "dominant_edges_after_all_stabilization": len(attachment_edges_after),
            "sample_before": attachment_edges_before[:64],
            "sample_after_stabilization": attachment_edges_after_stabilization[:64],
            "sample_after_same_chain_smoothing": (
                attachment_edges_after_same_chain_smoothing[:64]
            ),
            "sample_after_all_stabilization": attachment_edges_after[:64],
            "topology_deleted": False,
            "attachment_vertices_bound_toward_common_torso_ancestor": True,
            "semantic_chain_bounded_graph_expansion": True,
        },
        "hard_joint_seams": {
            "count_before": len(seams_before),
            "maximum_l1_before": seams_before[0]["weight_l1"],
            "sample_before": seams_before[:64],
            "affected_vertices": len(affected),
            "maximum_graph_ring": max(affected.values(), default=0),
            "count_after": len(seams_after),
            "maximum_l1_after": (
                seams_after[0]["weight_l1"] if seams_after else 0.0
            ),
            "sample_after": seams_after[:64],
        },
        "weight_change": {
            "changed_vertices": int(np.count_nonzero(delta > 1.0e-8)),
            "mean_l1_per_vertex": float(delta.mean()),
            "maximum_l1_per_vertex": float(delta.max(initial=0.0)),
        },
        "weights_before": weight_stats(weights_before, args.minimum_weight),
        "weights_after": weight_stats(final, args.minimum_weight),
        "animation_exported": False,
        "native_mesh_topology_preserved": True,
        "pbr_material_preserved": True,
        "fitted_skeleton_rest_matrices_preserved": True,
        "only_vertex_weights_modified": True,
        "status": "research_candidate_pending_retarget_and_dynamic_qa",
        "formal_dataset_registration_authorized": False,
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_QUADRUPED_JOINT_WEIGHT_SMOOTHING_OK "
        f"seams={len(seams_before)}->{len(seams_after)} "
        f"affected={len(affected)} output={output}"
    )


if __name__ == "__main__":
    main()
