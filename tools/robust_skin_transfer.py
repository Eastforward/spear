"""Geometry-aware helpers for transferring skeletal weights between meshes.

The helpers in this file deliberately avoid Blender APIs.  Blender scripts can
extract vertices/faces/vertex-group weights, call this module, then write the
result back to vertex groups.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq

import numpy as np


REGION_TAIL = 0
REGION_TORSO = 1
REGION_HEAD = 2
REGION_FRONT_LEFT_LEG = 3
REGION_FRONT_RIGHT_LEG = 4
REGION_HIND_LEFT_LEG = 5
REGION_HIND_RIGHT_LEG = 6

REGION_NAMES = {
    REGION_TAIL: "tail",
    REGION_TORSO: "torso",
    REGION_HEAD: "head",
    REGION_FRONT_LEFT_LEG: "front_left_leg",
    REGION_FRONT_RIGHT_LEG: "front_right_leg",
    REGION_HIND_LEFT_LEG: "hind_left_leg",
    REGION_HIND_RIGHT_LEG: "hind_right_leg",
}


@dataclass(frozen=True)
class SkeletonCapsule:
    """A semantic bone segment used for instance-level target segmentation."""

    region: int
    start: object
    end: object
    radius: float = 0.0


def normalize_rows(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a row-normalized copy; all-zero rows remain all-zero."""
    out = np.asarray(weights, dtype=np.float64).copy()
    sums = out.sum(axis=1, keepdims=True)
    ok = sums[:, 0] > eps
    out[ok] /= sums[ok]
    out[~ok] = 0.0
    return out


def keep_top_k_normalized(weights: np.ndarray, k: int = 4) -> np.ndarray:
    """Keep only the top-k influences in each row, then renormalize."""
    out = np.asarray(weights, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError("weights must be a 2-D array")
    if k <= 0:
        return np.zeros_like(out)
    if k < out.shape[1]:
        keep = np.argpartition(out, -k, axis=1)[:, -k:]
        mask = np.zeros_like(out, dtype=bool)
        rows = np.arange(out.shape[0])[:, None]
        mask[rows, keep] = True
        out[~mask] = 0.0
    return normalize_rows(out)


def mesh_bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices.min(axis=0), vertices.max(axis=0)


def _normalized_xyz(
    vertices: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    mn, mx = bounds if bounds is not None else mesh_bounds(vertices)
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    extent = np.maximum(mx - mn, 1e-12)
    return (vertices - mn) / extent


def coarse_region_labels(
    vertices: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    tail_x: float = 0.16,
    head_x: float = 0.64,
    leg_y: float = 0.36,
    front_x: float = 0.50,
) -> np.ndarray:
    """Label dog-like mesh points in source-rig coordinates.

    Coordinate convention for this asset family is +X=head/front, +Y=up, and
    Z=lateral.  These are deliberately coarse regions: they prevent a head or
    tail vertex from stealing weights from a nearby but semantically unrelated
    leg face, then the inpainting pass smooths small gaps.
    """
    uvw = _normalized_xyz(vertices, bounds)
    x = uvw[:, 0]
    y = uvw[:, 1]
    z = uvw[:, 2]
    labels = np.full(len(vertices), REGION_TORSO, dtype=np.int64)

    labels[x <= tail_x] = REGION_TAIL
    labels[(x >= head_x) & (y >= leg_y)] = REGION_HEAD

    leg_mask = (y < leg_y) & (x > tail_x)
    left = z >= 0.5
    front = x >= front_x
    labels[leg_mask & front & left] = REGION_FRONT_LEFT_LEG
    labels[leg_mask & front & ~left] = REGION_FRONT_RIGHT_LEG
    labels[leg_mask & ~front & left] = REGION_HIND_LEFT_LEG
    labels[leg_mask & ~front & ~left] = REGION_HIND_RIGHT_LEG
    return labels


def face_region_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    centers = vertices[faces].mean(axis=1)
    return coarse_region_labels(centers, bounds=bounds)


def build_vertex_adjacency(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    neighbors = [set() for _ in range(n_vertices)]
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = [int(i) for i in tri]
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(n) for n in neighbors]


def regularize_regions_by_connected_components(
    *,
    faces: np.ndarray,
    labels: np.ndarray,
    eligible_regions: set[int] | tuple[int, ...] | list[int] | None = None,
    vote_bias: dict[int, float] | None = None,
    min_component_size: int = 1,
) -> tuple[np.ndarray, dict[str, int]]:
    """Make disconnected target shells use one semantic region when eligible."""
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    n_vertices = len(labels)
    out = labels.copy()
    if n_vertices == 0:
        return out, {"components": 0, "regularized_components": 0, "changed_vertices": 0}

    parent = list(range(n_vertices))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for tri in faces:
        if len(tri) < 3:
            continue
        a = int(tri[0])
        for b in tri[1:]:
            union(a, int(b))

    components: dict[int, list[int]] = {}
    for vertex_index in range(n_vertices):
        components.setdefault(find(vertex_index), []).append(vertex_index)

    eligible = None if eligible_regions is None else {int(region) for region in eligible_regions}
    vote_bias = {} if vote_bias is None else {int(region): float(bias) for region, bias in vote_bias.items()}
    regularized_components = 0
    changed_vertices = 0
    skipped_components = 0
    for component in components.values():
        if len(component) < min_component_size:
            skipped_components += 1
            continue
        component_labels = labels[component]
        unique_regions = {int(region) for region in component_labels}
        if eligible is not None and not unique_regions.issubset(eligible):
            skipped_components += 1
            continue
        regions, counts = np.unique(component_labels, return_counts=True)
        if len(regions) <= 1:
            continue
        scores = np.asarray([
            float(count) * vote_bias.get(int(region), 1.0)
            for region, count in zip(regions, counts)
        ])
        chosen = int(regions[int(np.argmax(scores))])
        changed = int(np.sum(out[component] != chosen))
        out[component] = chosen
        changed_vertices += changed
        regularized_components += 1

    return out, {
        "components": int(len(components)),
        "regularized_components": int(regularized_components),
        "changed_vertices": int(changed_vertices),
        "skipped_components": int(skipped_components),
    }


def _segment_distances(vertices: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    ab = end - start
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.linalg.norm(vertices - start, axis=1)
    t = np.clip(((vertices - start) @ ab) / denom, 0.0, 1.0)
    closest = start + t[:, None] * ab
    return np.linalg.norm(vertices - closest, axis=1)


def capsule_region_distances(
    vertices: np.ndarray,
    capsules: list[SkeletonCapsule],
    region_ids: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return min distance from each vertex to each semantic capsule region."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if region_ids is None:
        region_ids = sorted({int(c.region) for c in capsules})
    region_ids = np.asarray(region_ids, dtype=np.int64)
    region_to_col = {int(region): i for i, region in enumerate(region_ids)}
    distances = np.full((len(vertices), len(region_ids)), np.inf, dtype=np.float64)
    for capsule in capsules:
        region = int(capsule.region)
        if region not in region_to_col:
            continue
        start = np.asarray(capsule.start, dtype=np.float64)
        end = np.asarray(capsule.end, dtype=np.float64)
        d = np.maximum(0.0, _segment_distances(vertices, start, end) - float(capsule.radius))
        col = region_to_col[region]
        distances[:, col] = np.minimum(distances[:, col], d)
    return distances, region_ids


def graph_region_labels_from_capsules(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    capsules: list[SkeletonCapsule],
    seed_labels: np.ndarray | None = None,
    coarse_labels: np.ndarray | None = None,
    region_ids: list[int] | np.ndarray | None = None,
    unary_weight: float = 1.5,
    protected_region_unary_scale: float = 0.25,
    seed_distance_ratio: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    """Segment target mesh with semantic capsules and mesh connectivity.

    Seeds are hard anchors.  Labels then propagate over the target mesh graph,
    with a soft preference for staying near the corresponding semantic capsule.
    This lets a connected tail keep the tail label even when some tail vertices
    are geometrically closer to a hind-leg capsule.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = len(vertices)
    if n_vertices == 0:
        return np.zeros(0, dtype=np.int64), {"seed_count": 0}
    if not capsules:
        labels = coarse_region_labels(vertices)
        return labels, {"seed_count": 0, "fallback": "coarse_no_capsules"}

    if region_ids is None:
        region_ids = sorted({
            REGION_TAIL,
            REGION_TORSO,
            REGION_HEAD,
            REGION_FRONT_LEFT_LEG,
            REGION_FRONT_RIGHT_LEG,
            REGION_HIND_LEFT_LEG,
            REGION_HIND_RIGHT_LEG,
            *[int(c.region) for c in capsules],
        })
    capsule_distances, region_ids = capsule_region_distances(vertices, capsules, region_ids)
    region_to_col = {int(region): i for i, region in enumerate(region_ids)}
    finite = capsule_distances[np.isfinite(capsule_distances)]
    bounds = mesh_bounds(vertices)
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    scale = max(diag * 0.05, float(np.median(finite)) if finite.size else 1.0, 1e-9)
    unary = np.minimum(np.nan_to_num(capsule_distances / scale, nan=10.0, posinf=10.0), 10.0)
    for protected in (REGION_HEAD, REGION_TAIL):
        col = region_to_col.get(protected)
        if col is not None:
            unary[:, col] *= float(protected_region_unary_scale)

    auto_seed_from_capsules = seed_labels is None
    if seed_labels is None:
        seed_labels = np.full(n_vertices, -1, dtype=np.int64)
    else:
        seed_labels = np.asarray(seed_labels, dtype=np.int64).copy()
    seed_distance = max(diag * seed_distance_ratio, 1e-9)
    if coarse_labels is not None:
        coarse_labels = np.asarray(coarse_labels, dtype=np.int64)
        for protected in (REGION_HEAD, REGION_TAIL):
            col = region_to_col.get(protected)
            if col is None:
                continue
            close_to_capsule = np.isfinite(capsule_distances[:, col]) & (
                capsule_distances[:, col] <= seed_distance
            )
            seed_labels[(coarse_labels == protected) & close_to_capsule] = protected

    nearest_cols = np.argmin(unary, axis=1)
    nearest_regions = region_ids[nearest_cols]
    nearest_raw_dist = capsule_distances[np.arange(n_vertices), nearest_cols]
    if auto_seed_from_capsules:
        confident = (seed_labels < 0) & np.isfinite(nearest_raw_dist) & (nearest_raw_dist <= seed_distance)
        seed_labels[confident] = nearest_regions[confident]

    for col, region in enumerate(region_ids):
        if np.any(seed_labels == region):
            continue
        d = capsule_distances[:, col]
        if np.isfinite(d).any():
            seed_labels[int(np.nanargmin(d))] = int(region)

    adjacency = build_vertex_adjacency(faces, n_vertices)
    n_regions = len(region_ids)
    dist = np.full((n_vertices, n_regions), np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for vertex_index, region in enumerate(seed_labels):
        if int(region) not in region_to_col:
            continue
        col = region_to_col[int(region)]
        if dist[vertex_index, col] > 0.0:
            dist[vertex_index, col] = 0.0
            heapq.heappush(queue, (0.0, vertex_index, col))

    while queue:
        cur_cost, vertex_index, col = heapq.heappop(queue)
        if cur_cost != dist[vertex_index, col]:
            continue
        for neighbor in adjacency[vertex_index]:
            edge_len = float(np.linalg.norm(vertices[neighbor] - vertices[vertex_index]))
            step = edge_len * (1.0 + unary_weight * unary[neighbor, col])
            new_cost = cur_cost + step
            if new_cost < dist[neighbor, col]:
                dist[neighbor, col] = new_cost
                heapq.heappush(queue, (new_cost, neighbor, col))

    labels = np.empty(n_vertices, dtype=np.int64)
    reachable = np.isfinite(dist).any(axis=1)
    labels[reachable] = region_ids[np.argmin(dist[reachable], axis=1)]
    labels[~reachable] = nearest_regions[~reachable]
    seeded = seed_labels >= 0
    labels[seeded] = seed_labels[seeded]
    stats = {
        "seed_count": int(seeded.sum()),
        "capsule_regions": [REGION_NAMES.get(int(r), str(int(r))) for r in region_ids],
        "protected_region_unary_scale": float(protected_region_unary_scale),
        "unreached": int((~reachable).sum()),
        "region_counts": {
            REGION_NAMES.get(int(region), str(int(region))): int(np.sum(labels == region))
            for region in sorted(set(int(x) for x in labels))
        },
    }
    return labels, stats


def inpaint_missing_weights(
    faces: np.ndarray,
    weights: np.ndarray,
    known_mask: np.ndarray,
    *,
    max_iterations: int = 64,
    eps: float = 1e-12,
    fallback_to_global: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill unknown skin weights by graph-neighbor averaging.

    Known rows are treated as hard constraints.  Unknown rows become known once
    any adjacent vertex has a known/filled row.  If a disconnected component has
    no known rows, it can optionally receive the global known average.
    """
    out = normalize_rows(weights, eps=eps)
    known_mask = np.asarray(known_mask, dtype=bool)
    filled = known_mask & (out.sum(axis=1) > eps)
    adjacency = build_vertex_adjacency(faces, len(out))

    n_iters = 0
    for n_iters in range(1, max_iterations + 1):
        next_out = out.copy()
        next_filled = filled.copy()
        progress = False
        for i, neigh in enumerate(adjacency):
            if filled[i]:
                continue
            usable = [j for j in neigh if filled[j] and out[j].sum() > eps]
            if not usable:
                continue
            next_out[i] = out[usable].mean(axis=0)
            next_filled[i] = True
            progress = True
        out = normalize_rows(next_out, eps=eps)
        filled = next_filled
        if not progress or filled.all():
            break

    if fallback_to_global and not filled.all() and filled.any():
        global_avg = normalize_rows(out[filled].mean(axis=0, keepdims=True), eps=eps)[0]
        out[~filled] = global_avg
        filled[~filled] = True

    return normalize_rows(out, eps=eps), filled, n_iters


def compatible_region_candidates(region: int) -> tuple[int, ...]:
    """Return primary-compatible source regions, with conservative fallbacks."""
    region = int(region)
    if region == REGION_HEAD:
        return (REGION_HEAD, REGION_TORSO)
    if region == REGION_TAIL:
        return (REGION_TAIL, REGION_TORSO)
    if region in (
        REGION_FRONT_LEFT_LEG,
        REGION_FRONT_RIGHT_LEG,
        REGION_HIND_LEFT_LEG,
        REGION_HIND_RIGHT_LEG,
    ):
        return (region, REGION_TORSO)
    return (REGION_TORSO,)


def _closest_point_barycentric(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Closest-point barycentric coordinates on triangle abc."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        bary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return bary, float(np.linalg.norm(p - a))

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        bary = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return bary, float(np.linalg.norm(p - b))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        bary = np.array([1.0 - v, v, 0.0], dtype=np.float64)
        return bary, float(np.linalg.norm(p - q))

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        bary = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return bary, float(np.linalg.norm(p - c))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        bary = np.array([1.0 - w, 0.0, w], dtype=np.float64)
        return bary, float(np.linalg.norm(p - q))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * (c - b)
        bary = np.array([0.0, 1.0 - w, w], dtype=np.float64)
        return bary, float(np.linalg.norm(p - q))

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    q = u * a + v * b + w * c
    bary = np.array([u, v, w], dtype=np.float64)
    return bary, float(np.linalg.norm(p - q))


def _candidate_faces_for_region(
    source_face_regions: np.ndarray,
    region: int,
) -> np.ndarray:
    for candidate_region in compatible_region_candidates(int(region)):
        idx = np.flatnonzero(source_face_regions == candidate_region)
        if len(idx):
            return idx
    return np.arange(len(source_face_regions), dtype=np.int64)


def target_region_labels_from_source_proximity(
    *,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_face_regions: np.ndarray,
    target_vertices: np.ndarray,
    coarse_target_regions: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Infer target regions from the nearest source-rig face region.

    Pure spatial thresholding tends to label long belly fur as "leg" because it
    hangs low.  Source proximity is a better generic hint for body-vs-limb
    separation, while coarse head/tail labels remain protected so those ends do
    not get stolen by nearby torso/leg faces.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    source_face_regions = np.asarray(source_face_regions, dtype=np.int64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    if coarse_target_regions is None:
        coarse_target_regions = coarse_region_labels(target_vertices, bounds=bounds)
    coarse_target_regions = np.asarray(coarse_target_regions, dtype=np.int64)

    centers = source_vertices[source_faces].mean(axis=1)
    labels = np.empty(len(target_vertices), dtype=np.int64)
    for start in range(0, len(target_vertices), chunk_size):
        stop = min(start + chunk_size, len(target_vertices))
        chunk = target_vertices[start:stop]
        d2 = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:stop] = source_face_regions[np.argmin(d2, axis=1)]

    for protected in (REGION_HEAD, REGION_TAIL):
        protected_mask = coarse_target_regions == protected
        labels[protected_mask] = protected
        leak_mask = (labels == protected) & ~protected_mask
        labels[leak_mask] = REGION_TORSO
    return labels


def transfer_weights_by_region(
    *,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_weights: np.ndarray,
    target_vertices: np.ndarray,
    source_face_regions: np.ndarray | None = None,
    target_regions: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    max_distance: float | None = None,
    candidate_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Transfer weights by nearest compatible source triangle.

    Unlike an unconstrained closest-surface transfer, each target vertex first
    chooses source faces from the same coarse anatomical region.  This protects
    tail/head vertices from the classic nearest-but-wrong leg match.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    source_weights = normalize_rows(source_weights)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)

    if source_face_regions is None:
        source_face_regions = face_region_labels(source_vertices, source_faces, bounds=bounds)
    source_face_regions = np.asarray(source_face_regions, dtype=np.int64)
    if target_regions is None:
        target_regions = coarse_region_labels(target_vertices, bounds=bounds)
    target_regions = np.asarray(target_regions, dtype=np.int64)

    face_vertices = source_vertices[source_faces]
    face_centers = face_vertices.mean(axis=1)
    out = np.zeros((len(target_vertices), source_weights.shape[1]), dtype=np.float64)
    matched = np.zeros(len(target_vertices), dtype=bool)
    region_counts: dict[str, int] = defaultdict(int)
    no_compatible_faces = 0
    over_distance = 0

    face_cache: dict[int, np.ndarray] = {}
    for i, p in enumerate(target_vertices):
        region = int(target_regions[i])
        region_counts[REGION_NAMES.get(region, str(region))] += 1
        if region not in face_cache:
            face_cache[region] = _candidate_faces_for_region(source_face_regions, region)
        candidates = face_cache[region]
        if len(candidates) == 0:
            no_compatible_faces += 1
            continue

        center_d2 = np.sum((face_centers[candidates] - p) ** 2, axis=1)
        n_take = min(max(1, candidate_count), len(candidates))
        if n_take < len(candidates):
            local = np.argpartition(center_d2, n_take - 1)[:n_take]
        else:
            local = np.arange(len(candidates))

        best_dist = float("inf")
        best_weight = None
        for face_idx in candidates[local]:
            tri = face_vertices[int(face_idx)]
            bary, dist = _closest_point_barycentric(p, tri[0], tri[1], tri[2])
            if dist >= best_dist:
                continue
            src_face = source_faces[int(face_idx)]
            best_dist = dist
            best_weight = (
                bary[0] * source_weights[src_face[0]]
                + bary[1] * source_weights[src_face[1]]
                + bary[2] * source_weights[src_face[2]]
            )

        if best_weight is None:
            no_compatible_faces += 1
            continue
        if max_distance is not None and best_dist > max_distance:
            over_distance += 1
            continue

        out[i] = best_weight
        matched[i] = True

    stats = {
        "target_vertices": int(len(target_vertices)),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "no_compatible_faces": int(no_compatible_faces),
        "over_distance": int(over_distance),
        "region_counts": dict(region_counts),
    }
    return normalize_rows(out), matched, stats
