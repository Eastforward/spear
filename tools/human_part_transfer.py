"""Pure NumPy helpers for part-aware humanoid skin-weight transfer."""
from __future__ import annotations

from collections import Counter
from enum import IntEnum
from typing import Mapping, Sequence
import re

import numpy as np

from tools.robust_skin_transfer import (
    _closest_point_barycentric,
    ground_artifact_vertex_mask,
    keep_top_k_normalized,
    normalize_rows,
)


class HumanRegion(IntEnum):
    TORSO = 0
    HEAD = 1
    LEFT_UPPER_ARM = 2
    LEFT_FOREARM = 3
    LEFT_PALM = 4
    RIGHT_UPPER_ARM = 5
    RIGHT_FOREARM = 6
    RIGHT_PALM = 7
    LEFT_THIGH = 8
    LEFT_CALF = 9
    LEFT_FOOT = 10
    RIGHT_THIGH = 11
    RIGHT_CALF = 12
    RIGHT_FOOT = 13


_LEFT_REGIONS = {
    HumanRegion.LEFT_UPPER_ARM,
    HumanRegion.LEFT_FOREARM,
    HumanRegion.LEFT_PALM,
    HumanRegion.LEFT_THIGH,
    HumanRegion.LEFT_CALF,
    HumanRegion.LEFT_FOOT,
}
_RIGHT_REGIONS = {
    HumanRegion.RIGHT_UPPER_ARM,
    HumanRegion.RIGHT_FOREARM,
    HumanRegion.RIGHT_PALM,
    HumanRegion.RIGHT_THIGH,
    HumanRegion.RIGHT_CALF,
    HumanRegion.RIGHT_FOOT,
}
_CENTER_REGIONS = {HumanRegion.TORSO, HumanRegion.HEAD}
_LEFT_LEG_REGIONS = {
    HumanRegion.LEFT_THIGH,
    HumanRegion.LEFT_CALF,
    HumanRegion.LEFT_FOOT,
}
_RIGHT_LEG_REGIONS = {
    HumanRegion.RIGHT_THIGH,
    HumanRegion.RIGHT_CALF,
    HumanRegion.RIGHT_FOOT,
}

_COMPATIBLE_REGIONS = {
    HumanRegion.TORSO: {
        HumanRegion.TORSO,
        HumanRegion.HEAD,
        HumanRegion.LEFT_UPPER_ARM,
        HumanRegion.RIGHT_UPPER_ARM,
        HumanRegion.LEFT_THIGH,
        HumanRegion.RIGHT_THIGH,
    },
    HumanRegion.HEAD: {HumanRegion.HEAD, HumanRegion.TORSO},
    HumanRegion.LEFT_UPPER_ARM: {
        HumanRegion.LEFT_UPPER_ARM,
        HumanRegion.LEFT_FOREARM,
        HumanRegion.TORSO,
    },
    HumanRegion.LEFT_FOREARM: {
        HumanRegion.LEFT_UPPER_ARM,
        HumanRegion.LEFT_FOREARM,
        HumanRegion.LEFT_PALM,
    },
    HumanRegion.LEFT_PALM: {HumanRegion.LEFT_FOREARM, HumanRegion.LEFT_PALM},
    HumanRegion.RIGHT_UPPER_ARM: {
        HumanRegion.RIGHT_UPPER_ARM,
        HumanRegion.RIGHT_FOREARM,
        HumanRegion.TORSO,
    },
    HumanRegion.RIGHT_FOREARM: {
        HumanRegion.RIGHT_UPPER_ARM,
        HumanRegion.RIGHT_FOREARM,
        HumanRegion.RIGHT_PALM,
    },
    HumanRegion.RIGHT_PALM: {HumanRegion.RIGHT_FOREARM, HumanRegion.RIGHT_PALM},
    HumanRegion.LEFT_THIGH: {
        HumanRegion.LEFT_THIGH,
        HumanRegion.LEFT_CALF,
        HumanRegion.TORSO,
    },
    HumanRegion.LEFT_CALF: {
        HumanRegion.LEFT_THIGH,
        HumanRegion.LEFT_CALF,
        HumanRegion.LEFT_FOOT,
    },
    HumanRegion.LEFT_FOOT: {HumanRegion.LEFT_CALF, HumanRegion.LEFT_FOOT},
    HumanRegion.RIGHT_THIGH: {
        HumanRegion.RIGHT_THIGH,
        HumanRegion.RIGHT_CALF,
        HumanRegion.TORSO,
    },
    HumanRegion.RIGHT_CALF: {
        HumanRegion.RIGHT_THIGH,
        HumanRegion.RIGHT_CALF,
        HumanRegion.RIGHT_FOOT,
    },
    HumanRegion.RIGHT_FOOT: {HumanRegion.RIGHT_CALF, HumanRegion.RIGHT_FOOT},
}


def _name_tokens(name: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(name).lower())


def _bone_side(tokens: Sequence[str]) -> str | None:
    if "l" in tokens or "left" in tokens:
        return "left"
    if "r" in tokens or "right" in tokens:
        return "right"
    return None


def _group_region(name: object) -> HumanRegion:
    tokens = _name_tokens(name)
    joined = " ".join(tokens)
    side = _bone_side(tokens)

    facial_markers = (
        "eye",
        "jaw",
        "lip",
        "mouth",
        "cheek",
        "eyebrow",
        "nose",
        "tongue",
        "masseter",
        "caninus",
    )
    if any(marker in joined for marker in facial_markers):
        return HumanRegion.HEAD

    if "finger" in joined or "hand" in tokens or "palm" in tokens:
        if side == "left":
            return HumanRegion.LEFT_PALM
        if side == "right":
            return HumanRegion.RIGHT_PALM
    if "forearm" in joined or "lowerarm" in joined:
        if side == "left":
            return HumanRegion.LEFT_FOREARM
        if side == "right":
            return HumanRegion.RIGHT_FOREARM
    if "upperarm" in joined or ("upper" in tokens and "arm" in tokens):
        if side == "left":
            return HumanRegion.LEFT_UPPER_ARM
        if side == "right":
            return HumanRegion.RIGHT_UPPER_ARM
    if "clavicle" in joined:
        if side == "left":
            return HumanRegion.LEFT_UPPER_ARM
        if side == "right":
            return HumanRegion.RIGHT_UPPER_ARM
    if "thigh" in joined or "upleg" in joined:
        if side == "left":
            return HumanRegion.LEFT_THIGH
        if side == "right":
            return HumanRegion.RIGHT_THIGH
    if "calf" in joined or "shin" in joined or "lowerleg" in joined:
        if side == "left":
            return HumanRegion.LEFT_CALF
        if side == "right":
            return HumanRegion.RIGHT_CALF
    if "foot" in joined or "toe" in joined:
        if side == "left":
            return HumanRegion.LEFT_FOOT
        if side == "right":
            return HumanRegion.RIGHT_FOOT
    if "head" in joined or "neck" in joined:
        return HumanRegion.HEAD
    return HumanRegion.TORSO


def source_vertex_regions_from_weights(
    weights: np.ndarray,
    group_names: Sequence[object],
) -> np.ndarray:
    """Classify source vertices by summed vertex-group mass per human region."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D array")
    if weights.shape[1] != len(group_names):
        raise ValueError("group_names must match the weight columns")

    mass = np.zeros((len(weights), len(HumanRegion)), dtype=np.float64)
    positive_weights = np.maximum(weights, 0.0)
    for column, name in enumerate(group_names):
        mass[:, int(_group_region(name))] += positive_weights[:, column]
    return np.argmax(mass, axis=1).astype(np.int64)


def source_face_regions(faces: np.ndarray, vertex_regions: np.ndarray) -> np.ndarray:
    """Assign each face the majority region of its source vertices."""
    faces = np.asarray(faces, dtype=np.int64)
    vertex_regions = np.asarray(vertex_regions, dtype=np.int64)
    out = np.empty(len(faces), dtype=np.int64)
    for index, face in enumerate(faces):
        if len(face) == 0:
            out[index] = int(HumanRegion.TORSO)
            continue
        counts = Counter(int(vertex_regions[int(vertex)]) for vertex in face)
        out[index] = min(counts, key=lambda region: (-counts[region], region))
    return out


def _segment_distance(vertices: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-20:
        return np.linalg.norm(vertices - start, axis=1)
    t = np.sum((vertices - start) * direction, axis=1) / length_squared
    closest = start + np.clip(t, 0.0, 1.0)[:, None] * direction
    return np.linalg.norm(vertices - closest, axis=1)


def _capsule_records(capsules: object) -> list[tuple[HumanRegion, np.ndarray, np.ndarray, float]]:
    records: list[tuple[HumanRegion, np.ndarray, np.ndarray, float]] = []
    items = capsules.items() if isinstance(capsules, Mapping) else capsules
    for item in items:
        if isinstance(capsules, Mapping):
            region, capsule = item
            values = capsule
            if len(values) not in (2, 3):
                raise ValueError("capsules must contain (start, end[, radius]) tuples")
            start, end = values[:2]
            radius = values[2] if len(values) == 3 else 0.0
        elif hasattr(item, "region"):
            region = item.region
            start = item.start
            end = item.end
            radius = getattr(item, "radius", 0.0)
        else:
            if len(item) not in (3, 4):
                raise ValueError("capsules must contain (region, start, end[, radius]) tuples")
            region, start, end = item[:3]
            radius = item[3] if len(item) == 4 else 0.0
        records.append(
            (
                HumanRegion(int(region)),
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
                float(radius),
            )
        )
    if not records:
        raise ValueError("at least one capsule is required")
    return records


def target_regions_from_capsules(vertices: np.ndarray, capsules: object) -> np.ndarray:
    """Label targets by nearest rest-bone capsule with geometric side eligibility."""
    vertices = np.asarray(vertices, dtype=np.float64)
    records = _capsule_records(capsules)
    scores = np.empty((len(vertices), len(records)), dtype=np.float64)
    for column, (_, start, end, radius) in enumerate(records):
        scores[:, column] = _segment_distance(vertices, start, end) - max(radius, 0.0)

    record_regions = [record[0] for record in records]
    left_columns = [index for index, region in enumerate(record_regions) if region in _LEFT_REGIONS]
    right_columns = [index for index, region in enumerate(record_regions) if region in _RIGHT_REGIONS]
    center_columns = [index for index, region in enumerate(record_regions) if region in _CENTER_REGIONS]

    labels = np.empty(len(vertices), dtype=np.int64)
    for row in range(len(vertices)):
        eligible = list(center_columns)
        left_score = min((scores[row, index] for index in left_columns), default=float("inf"))
        right_score = min((scores[row, index] for index in right_columns), default=float("inf"))
        eligible.extend(left_columns if left_score <= right_score else right_columns)
        if not eligible:
            eligible = list(range(len(records)))
        best = min(eligible, key=lambda index: (scores[row, index], index))
        labels[row] = int(record_regions[best])
    return labels


def _allowed_group_mask(group_regions: np.ndarray, target_region: int) -> np.ndarray:
    compatible = _COMPATIBLE_REGIONS[HumanRegion(int(target_region))]
    return np.isin(group_regions, [int(region) for region in compatible])


def _opposite_sides(first: int, second: int) -> bool:
    first_region = HumanRegion(int(first))
    second_region = HumanRegion(int(second))
    return (
        first_region in _LEFT_REGIONS and second_region in _RIGHT_REGIONS
    ) or (
        first_region in _RIGHT_REGIONS and second_region in _LEFT_REGIONS
    )


def _region_aware_inpaint(
    faces: np.ndarray,
    weights: np.ndarray,
    known_mask: np.ndarray,
    target_regions: np.ndarray,
    group_regions: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill topology gaps without crossing side or target-region boundaries."""
    faces = np.asarray(faces, dtype=np.int64)
    target_regions = np.asarray(target_regions, dtype=np.int64)
    out = normalize_rows(weights)
    filled = np.asarray(known_mask, dtype=bool).copy() & (out.sum(axis=1) > 0.0)
    adjacency = [set() for _ in range(len(out))]

    for face in faces:
        face_regions = target_regions[face]
        has_left = np.isin(face_regions, [int(region) for region in _LEFT_REGIONS]).any()
        has_right = np.isin(face_regions, [int(region) for region in _RIGHT_REGIONS]).any()
        if has_left and has_right:
            continue
        for local_index, vertex in enumerate(face):
            for neighbor in face[local_index + 1 :]:
                adjacency[int(vertex)].add(int(neighbor))
                adjacency[int(neighbor)].add(int(vertex))

    n_iterations = 0
    for n_iterations in range(1, max(0, int(max_iterations)) + 1):
        next_out = out.copy()
        next_filled = filled.copy()
        progress = False
        for vertex, neighbors in enumerate(adjacency):
            if filled[vertex]:
                continue
            target_region = HumanRegion(int(target_regions[vertex]))
            compatible = _COMPATIBLE_REGIONS[target_region]
            usable = [
                neighbor
                for neighbor in neighbors
                if filled[neighbor]
                and not _opposite_sides(target_region, int(target_regions[neighbor]))
                and HumanRegion(int(target_regions[neighbor])) in compatible
            ]
            if not usable:
                continue
            candidate = out[usable].mean(axis=0)
            candidate[~_allowed_group_mask(group_regions, int(target_region))] = 0.0
            candidate = normalize_rows(candidate[None, :])[0]
            if candidate.sum() <= 0.0:
                continue
            next_out[vertex] = candidate
            next_filled[vertex] = True
            progress = True
        out = next_out
        filled = next_filled
        if not progress or filled.all():
            break
    return normalize_rows(out), filled, n_iterations


def collapse_finger_weights_to_palms(
    weights: np.ndarray,
    group_names: Sequence[object],
) -> np.ndarray:
    """Move every Rocketbox finger influence into the same-side Hand group."""
    out = np.asarray(weights, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != len(group_names):
        raise ValueError("group_names must match a 2-D weight array")

    tokens = [_name_tokens(name) for name in group_names]
    for side in ("left", "right"):
        hand_columns = [
            index
            for index, parts in enumerate(tokens)
            if _bone_side(parts) == side and "hand" in parts and not any("finger" in p for p in parts)
        ]
        finger_columns = [
            index
            for index, parts in enumerate(tokens)
            if _bone_side(parts) == side and any("finger" in part for part in parts)
        ]
        if not finger_columns:
            continue
        if not hand_columns:
            raise ValueError(f"finger groups require a matching {side} Hand group")
        hand = hand_columns[0]
        out[:, hand] += out[:, finger_columns].sum(axis=1)
        out[:, finger_columns] = 0.0
    return out


def transfer_human_weights(
    source: Mapping[str, object],
    target: Mapping[str, object],
    *,
    max_distance: float | None = None,
    candidate_count: int = 24,
    top_k: int = 4,
    max_fill_iterations: int = 64,
    require_complete: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Transfer weights through side-safe human regions and closest triangles."""
    source_vertices = np.asarray(source["vertices"], dtype=np.float64)
    source_faces_array = np.asarray(source["faces"], dtype=np.int64)
    source_weights = normalize_rows(np.asarray(source["weights"], dtype=np.float64))
    group_names = list(source["group_names"])
    target_vertices = np.asarray(target["vertices"], dtype=np.float64)
    target_faces = np.asarray(target.get("faces", np.empty((0, 3))), dtype=np.int64)

    vertex_regions = source_vertex_regions_from_weights(source_weights, group_names)
    face_regions = source_face_regions(source_faces_array, vertex_regions)
    if "regions" in target:
        target_regions = np.asarray(target["regions"], dtype=np.int64)
    elif "capsules" in target:
        target_regions = target_regions_from_capsules(target_vertices, target["capsules"])
    else:
        raise ValueError("target must provide regions or capsules")
    if len(target_regions) != len(target_vertices):
        raise ValueError("target regions must match target vertices")

    group_regions = np.array([int(_group_region(name)) for name in group_names], dtype=np.int64)
    face_vertices = source_vertices[source_faces_array]
    face_centers = face_vertices.mean(axis=1) if len(source_faces_array) else np.empty((0, 3))
    out = np.zeros((len(target_vertices), source_weights.shape[1]), dtype=np.float64)
    matched = np.zeros(len(target_vertices), dtype=bool)

    candidate_cache: dict[int, np.ndarray] = {}
    for target_index, point in enumerate(target_vertices):
        region = HumanRegion(int(target_regions[target_index]))
        if int(region) not in candidate_cache:
            compatible = [int(item) for item in _COMPATIBLE_REGIONS[region]]
            candidate_cache[int(region)] = np.flatnonzero(np.isin(face_regions, compatible))
        candidates = candidate_cache[int(region)]
        if len(candidates) == 0:
            continue

        center_distances = np.sum((face_centers[candidates] - point) ** 2, axis=1)
        take = min(max(1, int(candidate_count)), len(candidates))
        if take < len(candidates):
            local = np.argpartition(center_distances, take - 1)[:take]
            candidates = candidates[local]

        best_distance = float("inf")
        best_weights = None
        for face_index in candidates:
            triangle = face_vertices[int(face_index)]
            barycentric, distance = _closest_point_barycentric(
                point,
                triangle[0],
                triangle[1],
                triangle[2],
            )
            if distance >= best_distance:
                continue
            source_face = source_faces_array[int(face_index)]
            best_distance = distance
            best_weights = np.sum(
                source_weights[source_face] * barycentric[:, None],
                axis=0,
            )
        if best_weights is None or (max_distance is not None and best_distance > max_distance):
            continue

        best_weights[~_allowed_group_mask(group_regions, int(region))] = 0.0
        best_weights = normalize_rows(best_weights[None, :])[0]
        if best_weights.sum() > 0.0:
            out[target_index] = best_weights
            matched[target_index] = True

    initial_unmatched = int((~matched).sum())
    if initial_unmatched and matched.any():
        out, filled, _ = _region_aware_inpaint(
            target_faces,
            out,
            matched,
            target_regions,
            group_regions,
            max_iterations=max_fill_iterations,
        )
        matched = filled
        for index, region in enumerate(target_regions):
            out[index, ~_allowed_group_mask(group_regions, int(region))] = 0.0
        out = normalize_rows(out)
        matched &= out.sum(axis=1) > 0.0

    out = collapse_finger_weights_to_palms(out, group_names)
    out = keep_top_k_normalized(out, k=top_k)
    matched = out.sum(axis=1) > 0.0
    counts = Counter(HumanRegion(int(region)).name.lower() for region in target_regions)
    unmatched_count = int((~matched).sum())
    unmatched_regions = Counter(
        HumanRegion(int(region)).name.lower() for region in target_regions[~matched]
    )
    stats = {
        "target_vertices": int(len(target_vertices)),
        "initial_matched": int(len(target_vertices) - initial_unmatched),
        "initial_unmatched": initial_unmatched,
        "matched": int(matched.sum()),
        "unmatched": unmatched_count,
        "unmatched_regions": dict(unmatched_regions),
        "graph_filled": int(matched.sum() - (len(target_vertices) - initial_unmatched)),
        "region_counts": dict(counts),
    }
    if require_complete and unmatched_count:
        raise ValueError(
            f"incomplete human weight transfer: {unmatched_count} unmatched "
            f"target vertices; regions={dict(unmatched_regions)}"
        )
    return out, stats


def human_ground_artifact_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_regions: np.ndarray | None = None,
    min_foot_fraction: float = 0.5,
    **kwargs: object,
) -> np.ndarray:
    """Flag flat ground cards while preserving foot-dominant shoe components."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    artifact = ground_artifact_vertex_mask(vertices=vertices, faces=faces, **kwargs)
    if vertex_regions is None:
        return artifact

    vertex_regions = np.asarray(vertex_regions, dtype=np.int64)
    if len(vertex_regions) != len(vertices):
        raise ValueError("vertex_regions must match vertices")
    if not 0.0 <= float(min_foot_fraction) <= 1.0:
        raise ValueError("min_foot_fraction must be between zero and one")

    parent = np.arange(len(vertices), dtype=np.int64)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for face in faces:
        if len(face) < 2:
            continue
        first = int(face[0])
        for vertex in face[1:]:
            union(first, int(vertex))

    components: dict[int, list[int]] = {}
    for vertex in range(len(vertices)):
        components.setdefault(find(vertex), []).append(vertex)
    foot_regions = [int(HumanRegion.LEFT_FOOT), int(HumanRegion.RIGHT_FOOT)]
    for component in components.values():
        if not artifact[component].any():
            continue
        foot_fraction = float(np.mean(np.isin(vertex_regions[component], foot_regions)))
        if foot_fraction >= float(min_foot_fraction):
            artifact[component] = False
    return artifact


def cross_limb_bridge_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_regions: np.ndarray,
    *,
    pelvis_height: float,
    up_axis: int = 2,
) -> np.ndarray:
    """Flag below-pelvis faces containing vertices from both left and right legs."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_regions = np.asarray(vertex_regions, dtype=np.int64)
    bridge = np.zeros(len(faces), dtype=bool)
    if len(vertices) == 0 or len(faces) == 0:
        return bridge
    if len(vertex_regions) != len(vertices):
        raise ValueError("vertex_regions must match vertices")

    left = [int(region) for region in _LEFT_LEG_REGIONS]
    right = [int(region) for region in _RIGHT_LEG_REGIONS]
    limbs = left + right
    for face_index, face in enumerate(faces):
        if vertices[face, up_axis].max() >= float(pelvis_height):
            continue
        regions = vertex_regions[face]
        if (
            np.isin(regions, limbs).all()
            and np.isin(regions, left).any()
            and np.isin(regions, right).any()
        ):
            bridge[face_index] = True
    return bridge
