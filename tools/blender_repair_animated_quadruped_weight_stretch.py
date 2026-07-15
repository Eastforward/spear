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
import itertools
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
    LIMB_CHAINS,
    bone_records,
)
from tools.blender_smooth_generated_quadruped_joint_weights import (  # noqa: E402
    cross_limb_weight_count,
    dominant_semantics,
    extract_weights,
    install_weights,
    linked_armatures,
    lowest_common_ancestor_name,
    real_meshes,
    remove_cross_limb_leakage,
    top_k_normalize,
)
from tools.generated_quadruped_semantics import (  # noqa: E402
    infer_quadruped_semantics,
    quadruped_semantic_labels,
)


SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v2"


class ExplicitSemantics:
    def __init__(self, root, chains):
        self.root = root
        self._chains = {name: tuple(values) for name, values in chains.items()}

    def chains(self):
        return dict(self._chains)


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
    parser.add_argument(
        "--semantic-label-map",
        type=Path,
        help=(
            "Authenticated explicit bone-to-locomotion labels for trusted "
            "multi-root template rigs whose detached hoof chains are real "
            "weight-bearing components rather than disposable controls."
        ),
    )
    parser.add_argument(
        "--cross-limb-authority",
        choices=("largest-bone", "nearest-rest-chain", "low-slice-components"),
        default="largest-bone",
        help=(
            "How a vertex's unique locomotion-limb domain is selected. "
            "nearest-rest-chain uses immutable rest-pose bone segments and "
            "is the strict option for close left/right generated limbs."
        ),
    )
    parser.add_argument(
        "--limb-slice-height-fraction",
        type=float,
        default=0.4,
        help=(
            "For low-slice-components, keep the four largest disconnected "
            "components below this robust rest-height fraction."
        ),
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
    parser.add_argument(
        "--skip-cross-limb-preclean",
        action="store_true",
        help=(
            "Residual-pass mode for an already repaired asset. Keep the input "
            "weights intact before motion-aware seed repair; repeated semantic "
            "preclean is not assumed to be idempotent."
        ),
    )
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


def explicit_semantic_labels(path, armature, front_axis):
    path = require_input(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid explicit semantic label map: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "avengine_explicit_quadruped_semantic_labels_v1"
        or payload.get("front_axis") != front_axis
        or not isinstance(payload.get("labels"), dict)
        or not isinstance(payload.get("chains"), dict)
    ):
        raise RuntimeError("explicit semantic label map contract changed")
    bone_names = set(armature.data.bones.keys())
    labels = {str(name): str(value) for name, value in payload["labels"].items()}
    if set(labels) != bone_names:
        raise RuntimeError(
            "explicit semantic labels must cover the complete skeleton: "
            f"missing={sorted(bone_names - set(labels))} "
            f"extra={sorted(set(labels) - bone_names)}"
        )
    root = payload.get("root")
    if root not in bone_names:
        raise RuntimeError("explicit semantic root is missing from the armature")
    allowed = {
        "axial",
        "head",
        "tail",
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    }
    if not set(labels.values()).issubset(allowed):
        raise RuntimeError(
            f"explicit semantic labels contain unsupported values: {sorted(set(labels.values()) - allowed)}"
        )
    chains = {
        str(name): tuple(str(value) for value in values)
        for name, values in payload["chains"].items()
    }
    flattened = {name for values in chains.values() for name in values}
    if not flattened.issubset(bone_names):
        raise RuntimeError("explicit semantic chains reference missing bones")
    return ExplicitSemantics(root, chains), labels, {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": payload["schema"],
    }


def semantic_labels(armature, rest_vertices, front_axis, explicit_map=None):
    if explicit_map is not None:
        return explicit_semantic_labels(explicit_map, armature, front_axis)
    records = bone_records(armature)
    semantics = infer_quadruped_semantics(
        records,
        bbox_min=rest_vertices.min(axis=0),
        bbox_extent=np.ptp(rest_vertices, axis=0),
        front_axis=front_axis,
    )
    labels = quadruped_semantic_labels(
        semantics,
        records,
        bbox_min=rest_vertices.min(axis=0),
        bbox_extent=np.ptp(rest_vertices, axis=0),
        front_axis=front_axis,
    )
    if set(labels) != set(armature.data.bones.keys()):
        raise RuntimeError("semantic decomposition does not cover the skeleton")
    return semantics, labels, None


def _world_point(matrix, point):
    homogeneous = np.asarray((*point, 1.0), dtype=np.float64)
    return (matrix @ homogeneous)[:3]


def _point_segment_distance_squared(points, start, end):
    direction = end - start
    denominator = float(direction @ direction)
    if denominator <= np.finfo(np.float64).eps:
        return np.square(points - start[None, :]).sum(axis=1)
    fraction = ((points - start[None, :]) @ direction) / denominator
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = start[None, :] + fraction[:, None] * direction[None, :]
    return np.square(points - nearest).sum(axis=1)


def rest_limb_segment_distances(rest_vertices, armature, semantics):
    chain_names = sorted(LIMB_CHAINS)
    semantic_chains = semantics.chains()
    missing = [name for name in chain_names if not semantic_chains.get(name)]
    if missing:
        raise RuntimeError(f"semantic rig has no rest segments for {missing}")
    armature_world = np.asarray(armature.matrix_world, dtype=np.float64)
    distances = np.full(
        (len(rest_vertices), len(chain_names)), np.inf, dtype=np.float64
    )
    nearest_bones = np.empty(
        (len(rest_vertices), len(chain_names)), dtype=object
    )
    nearest_bones[:] = None
    for chain_index, chain in enumerate(chain_names):
        for name in semantic_chains[chain]:
            bone = armature.data.bones.get(name)
            if bone is None:
                raise RuntimeError(f"semantic rest segment is missing: {name}")
            start = _world_point(armature_world, bone.head_local)
            end = _world_point(armature_world, bone.tail_local)
            candidate = _point_segment_distance_squared(rest_vertices, start, end)
            update = candidate < distances[:, chain_index]
            distances[update, chain_index] = candidate[update]
            nearest_bones[update, chain_index] = name
    if not np.isfinite(distances).all() or np.any(nearest_bones == None):  # noqa: E711
        raise RuntimeError("rest-space limb authority contains no finite segment")
    return chain_names, distances, nearest_bones


def nearest_rest_limb_authority(
    rest_vertices,
    armature,
    semantics,
    weights,
    bone_names,
    labels,
    minimum_weight,
):
    """Assign every meaningfully limb-weighted vertex to one rest-space limb.

    The assignment is independent of generated skin weights.  This matters for
    Bone Heat output where one physical hoof can have its largest individual
    influence on the correct side while most of its aggregate weight belongs
    to the opposite side.  The authority remains immutable through all later
    motion-aware repair passes.
    """
    chain_names, distances, nearest_bones = rest_limb_segment_distances(
        rest_vertices, armature, semantics
    )

    limb_indices = np.asarray(
        [
            index
            for index, name in enumerate(bone_names)
            if labels[name] in LIMB_CHAINS
        ],
        dtype=np.int64,
    )
    limb_mass = weights[:, limb_indices].sum(axis=1)
    active = limb_mass >= minimum_weight
    closest = np.argmin(distances, axis=1)
    authority = np.full(len(rest_vertices), "non_limb", dtype=object)
    authority[active] = np.asarray(chain_names, dtype=object)[closest[active]]
    fallback_bone = np.full(len(rest_vertices), None, dtype=object)
    rows = np.arange(len(rest_vertices), dtype=np.int64)
    fallback_bone[active] = nearest_bones[rows[active], closest[active]]
    counts = {
        chain: int(np.count_nonzero(authority == chain)) for chain in chain_names
    }
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(
            f"rest-space authority did not find all four limb domains: {counts}"
        )
    diagonal = float(np.linalg.norm(np.ptp(rest_vertices, axis=0)))
    active_distance = np.sqrt(distances[rows[active], closest[active]])
    return authority, fallback_bone, {
        "method": "nearest_rest_chain_segment",
        "immutable_through_motion_repair": True,
        "limb_weighted_vertex_count": int(np.count_nonzero(active)),
        "assignment_counts": counts,
        "median_nearest_distance_ratio_of_rest_diagonal": float(
            np.median(active_distance) / diagonal
        ),
        "maximum_nearest_distance_ratio_of_rest_diagonal": float(
            active_distance.max(initial=0.0) / diagonal
        ),
    }


def low_slice_component_limb_authority(
    rest_vertices,
    edges,
    armature,
    semantics,
    height_fraction,
):
    """Use four disconnected lower-limb topology components as authority.

    A generated watertight body is one global shell, but below the shoulder and
    hip attachments its four legs must remain four disconnected induced
    components.  Selecting those components prevents torso, belly, mane, or
    tail vertices from being forced into an arbitrary left/right limb domain.
    """
    robust_floor, robust_ceiling = np.percentile(
        rest_vertices[:, 2], (0.1, 99.9)
    )
    robust_height = float(robust_ceiling - robust_floor)
    if robust_height <= np.finfo(np.float64).eps:
        raise RuntimeError("rest mesh has no robust vertical extent")
    threshold = float(robust_floor + height_fraction * robust_height)
    selected = rest_vertices[:, 2] <= threshold
    parent = np.arange(len(rest_vertices), dtype=np.int64)
    rank = np.zeros(len(rest_vertices), dtype=np.int8)

    def find(value):
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    for first, second in edges:
        if selected[int(first)] and selected[int(second)]:
            union(first, second)
    # glTF duplicates the same physical vertex at UV/normal seams.  Weld those
    # copies only in the audit graph; geometry, UVs, and exported topology stay
    # untouched.
    position_representative = {}
    for vertex in np.flatnonzero(selected):
        key = tuple(float(value) for value in rest_vertices[vertex])
        representative = position_representative.setdefault(key, int(vertex))
        union(vertex, representative)
    grouped = {}
    for vertex in np.flatnonzero(selected):
        grouped.setdefault(find(vertex), []).append(int(vertex))
    minimum_component_size = max(64, int(round(len(rest_vertices) * 0.002)))
    components = [
        np.asarray(values, dtype=np.int64)
        for values in grouped.values()
        if len(values) >= minimum_component_size
    ]
    components.sort(key=len, reverse=True)
    if len(components) < 4:
        raise RuntimeError(
            "low rest slice does not contain four substantial disconnected "
            f"limbs: sizes={[len(value) for value in components]}"
        )
    selected_components = components[:4]
    fifth_size = len(components[4]) if len(components) > 4 else 0
    smallest_limb = len(selected_components[-1])
    if fifth_size >= 0.8 * smallest_limb:
        raise RuntimeError(
            "low rest slice has an ambiguous fifth component: "
            f"top_sizes={[len(value) for value in components[:6]]}"
        )

    chain_names, distances, nearest_bones = rest_limb_segment_distances(
        rest_vertices, armature, semantics
    )
    cost = np.asarray(
        [
            [float(np.median(distances[component, index])) for index in range(4)]
            for component in selected_components
        ],
        dtype=np.float64,
    )
    best_permutation = min(
        itertools.permutations(range(4)),
        key=lambda permutation: sum(
            cost[row, chain_index]
            for row, chain_index in enumerate(permutation)
        ),
    )
    authority = np.full(len(rest_vertices), "non_limb", dtype=object)
    fallback_bone = np.full(len(rest_vertices), None, dtype=object)
    component_records = []
    for component_index, (component, chain_index) in enumerate(
        zip(selected_components, best_permutation)
    ):
        chain = chain_names[chain_index]
        authority[component] = chain
        fallback_bone[component] = nearest_bones[component, chain_index]
        points = rest_vertices[component]
        component_records.append(
            {
                "rank_by_vertex_count": component_index,
                "semantic_chain": chain,
                "vertex_count": int(len(component)),
                "centroid": [float(value) for value in points.mean(axis=0)],
                "bbox_min": [float(value) for value in points.min(axis=0)],
                "bbox_max": [float(value) for value in points.max(axis=0)],
                "median_chain_distance": float(
                    np.sqrt(np.median(distances[component, chain_index]))
                ),
            }
        )
    counts = {
        chain: int(np.count_nonzero(authority == chain)) for chain in chain_names
    }
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"low-slice assignment missed a limb: {counts}")
    return authority, fallback_bone, {
        "method": "four_largest_disconnected_low_slice_components",
        "immutable_through_motion_repair": True,
        "blender_up_axis": "positive-z",
        "robust_floor": float(robust_floor),
        "robust_ceiling": float(robust_ceiling),
        "height_fraction": float(height_fraction),
        "slice_threshold": threshold,
        "minimum_component_size": minimum_component_size,
        "substantial_component_sizes_descending": [
            int(len(value)) for value in components[:12]
        ],
        "fifth_component_size": int(fifth_size),
        "assignment_counts": counts,
        "components": component_records,
    }


def project_weights_to_limb_authority(
    weights,
    authority,
    fallback_bone,
    bone_names,
    labels,
    minimum_weight,
):
    """Preserve total limb mass while forbidding every non-authority limb."""
    result = weights.copy()
    bone_index = {name: index for index, name in enumerate(bone_names)}
    limb_indices = np.asarray(
        [index for index, name in enumerate(bone_names) if labels[name] in LIMB_CHAINS],
        dtype=np.int64,
    )
    removed_entries = 0
    removed_mass = 0.0
    fallback_vertices = 0
    for chain in sorted(LIMB_CHAINS):
        vertices = np.flatnonzero(authority == chain)
        if not len(vertices):
            continue
        allowed = np.asarray(
            [
                index
                for index, name in enumerate(bone_names)
                if labels[name] == chain
            ],
            dtype=np.int64,
        )
        forbidden = np.asarray(
            [index for index in limb_indices if index not in set(allowed)],
            dtype=np.int64,
        )
        block = result[vertices].copy()
        total_limb_mass = block[:, limb_indices].sum(axis=1)
        allowed_values = block[:, allowed].copy()
        allowed_mass = allowed_values.sum(axis=1)
        forbidden_values = block[:, forbidden]
        removed_entries += int(np.count_nonzero(forbidden_values >= minimum_weight))
        removed_mass += float(forbidden_values.sum())
        block[:, limb_indices] = 0.0
        has_allowed = allowed_mass > np.finfo(np.float64).eps
        if np.any(has_allowed):
            block[np.ix_(has_allowed, allowed)] = (
                allowed_values[has_allowed]
                * (total_limb_mass[has_allowed] / allowed_mass[has_allowed])[:, None]
            )
        for local_index in np.flatnonzero(~has_allowed):
            vertex = int(vertices[local_index])
            name = fallback_bone[vertex]
            if name not in bone_index or labels[name] != chain:
                raise RuntimeError(
                    f"invalid fallback bone for vertex {vertex}: {name!r}"
                )
            block[local_index, bone_index[name]] = total_limb_mass[local_index]
            fallback_vertices += 1
        sums = block.sum(axis=1)
        if np.any(sums <= 0.0):
            raise RuntimeError("limb-domain projection created an unweighted vertex")
        block /= sums[:, None]
        result[vertices] = block
    return result, removed_entries, removed_mass, fallback_vertices


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
    if not 0.1 <= args.limb_slice_height_fraction <= 0.6:
        raise SystemExit("--limb-slice-height-fraction must be in [0.1, 0.6]")

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
    semantics, labels, explicit_semantic_authority = semantic_labels(
        armature,
        rest_vertices,
        args.front_axis,
        args.semantic_label_map,
    )
    bone_names = [bone.name for bone in armature.data.bones]
    weights_before = extract_weights(mesh, bone_names)
    spatial_authority = None
    fallback_bone = None
    spatial_authority_details = None
    if args.cross_limb_authority == "nearest-rest-chain":
        spatial_authority, fallback_bone, spatial_authority_details = (
            nearest_rest_limb_authority(
                rest_vertices,
                armature,
                semantics,
                weights_before,
                bone_names,
                labels,
                args.minimum_weight,
            )
        )
        dominant_chains = spatial_authority
    elif args.cross_limb_authority == "low-slice-components":
        spatial_authority, fallback_bone, spatial_authority_details = (
            low_slice_component_limb_authority(
                rest_vertices,
                edges,
                armature,
                semantics,
                args.limb_slice_height_fraction,
            )
        )
        dominant_chains = spatial_authority
    else:
        _dominant_bones, dominant_chains = dominant_semantics(
            weights_before, bone_names, labels
        )
    cross_limb_before = cross_limb_weight_count(
        weights_before,
        dominant_chains,
        bone_names,
        labels,
        args.minimum_weight,
    )
    if args.skip_cross_limb_preclean:
        weights = weights_before.copy()
        cross_limb_entries_removed = 0
        cross_limb_mass_removed = 0.0
    else:
        if spatial_authority is not None:
            (
                weights,
                cross_limb_entries_removed,
                cross_limb_mass_removed,
                projection_fallback_vertices,
            ) = project_weights_to_limb_authority(
                weights_before,
                spatial_authority,
                fallback_bone,
                bone_names,
                labels,
                args.minimum_weight,
            )
        else:
            weights, cross_limb_entries_removed, cross_limb_mass_removed = (
                remove_cross_limb_leakage(
                    weights_before, dominant_chains, bone_names, labels
                )
            )
            projection_fallback_vertices = 0
    if args.skip_cross_limb_preclean:
        projection_fallback_vertices = 0
    weights = top_k_normalize(weights, args.top_k, args.minimum_weight)
    if spatial_authority is not None:
        cleaned_dominant_chains = spatial_authority
    else:
        _dominant_bones, cleaned_dominant_chains = dominant_semantics(
            weights, bone_names, labels
        )
    cross_limb_after = cross_limb_weight_count(
        weights,
        cleaned_dominant_chains,
        bone_names,
        labels,
        args.minimum_weight,
    )
    install_weights(mesh, weights, bone_names, args.minimum_weight)
    bpy.context.view_layer.update()
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
        if spatial_authority is not None:
            (
                repaired,
                pass_projection_entries,
                pass_projection_mass,
                pass_projection_fallback_vertices,
            ) = project_weights_to_limb_authority(
                repaired,
                spatial_authority,
                fallback_bone,
                bone_names,
                labels,
                args.minimum_weight,
            )
            mode_details["post_repair_limb_projection"] = {
                "entries_removed": pass_projection_entries,
                "weight_mass_removed": pass_projection_mass,
                "fallback_vertices": pass_projection_fallback_vertices,
            }
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
    final_cross_limb = cross_limb_weight_count(
        weights,
        spatial_authority if spatial_authority is not None else dominant_semantics(
            weights, bone_names, labels
        )[1],
        bone_names,
        labels,
        args.minimum_weight,
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
            "explicit_semantic_authority": explicit_semantic_authority,
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
            "skip_cross_limb_preclean": bool(args.skip_cross_limb_preclean),
            "cross_limb_authority": args.cross_limb_authority,
            "limb_slice_height_fraction": args.limb_slice_height_fraction,
        },
        "cross_limb_preclean": {
            "entries_before": cross_limb_before[0],
            "weight_mass_before": cross_limb_before[1],
            "entries_removed": cross_limb_entries_removed,
            "weight_mass_removed": cross_limb_mass_removed,
            "entries_after": cross_limb_after[0],
            "weight_mass_after": cross_limb_after[1],
            "skipped_for_residual_pass": bool(args.skip_cross_limb_preclean),
            "projection_fallback_vertices": projection_fallback_vertices,
            "spatial_authority": spatial_authority_details,
            "final_forbidden_entries": final_cross_limb[0],
            "final_forbidden_weight_mass": final_cross_limb[1],
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
            if not len(final_seeds) and final_cross_limb[0] == 0
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
