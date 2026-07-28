#!/usr/bin/env python3
"""Pure geometry contracts for bounded same-mesh quadruped limb repair.

The Blender entry point deliberately imports only the rejected Pixal mesh.
This module keeps the spatial mask, source-surface tail segmentation, immutable
surface comparison, and low-slice dynamic gate testable without Blender so a
repair cannot quietly widen from two limb corridors into a body or tail
replacement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import struct
from typing import Any, Iterable

import numpy as np


SCHEMA = "avengine_pixal_same_mesh_mirrored_limb_repair_v1"
IMPLEMENTATION_CONTRACT = "bounded_local_surface_identity_v2"


@dataclass(frozen=True)
class RepairSpec:
    """Normalized Blender-space scope for one bounded quadruped repair."""

    source_side: str = "positive-y"
    head_direction: str = "negative-x"
    front_foot_x_fraction: float = 0.27
    front_attachment_x_fraction: float = 0.36
    hind_foot_x_fraction: float = 0.83
    hind_attachment_x_fraction: float = 0.68
    attachment_height_fraction: float = 0.58
    foot_half_width_fraction: float = 0.065
    attachment_half_width_fraction: float = 0.12
    source_side_guard_fraction: float = 0.01
    central_attachment_bridge_start_fraction: float = 0.38
    mirrored_attachment_taper_start_fraction: float = 0.34
    mirrored_attachment_top_lateral_scale: float = 0.35
    tail_protection_x_fraction: float = 0.78
    tail_protection_height_fraction: float = 0.38
    low_slice_height_fraction: float = 0.20

    def validate(self) -> None:
        if self.source_side not in {"positive-y", "negative-y"}:
            raise ValueError("source_side must be positive-y or negative-y")
        if self.head_direction not in {"positive-x", "negative-x"}:
            raise ValueError("head_direction must be positive-x or negative-x")
        fractions = {
            name: value
            for name, value in asdict(self).items()
            if name.endswith("_fraction")
        }
        if any(not 0.0 < float(value) < 1.0 for value in fractions.values()):
            raise ValueError(f"repair fractions must be in (0, 1): {fractions}")
        if self.foot_half_width_fraction >= self.attachment_half_width_fraction:
            raise ValueError("foot corridor must be narrower than attachment corridor")
        if not 0.0 < self.mirrored_attachment_top_lateral_scale <= 1.0:
            raise ValueError("mirrored attachment top lateral scale must be in (0, 1]")
        if (
            self.mirrored_attachment_taper_start_fraction
            >= self.attachment_height_fraction
        ):
            raise ValueError("mirrored attachment taper must start below attachment top")
        if self.head_direction == "negative-x":
            ordered = (
                self.front_foot_x_fraction
                < self.front_attachment_x_fraction
                < self.hind_attachment_x_fraction
                < self.hind_foot_x_fraction
            )
        else:
            ordered = (
                self.front_foot_x_fraction
                > self.front_attachment_x_fraction
                > self.hind_attachment_x_fraction
                > self.hind_foot_x_fraction
            )
        if not ordered:
            raise ValueError("front/hind foot and attachment fractions are inconsistent")

    def record(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("geometry points must be an Nx3 array with at least 8 rows")
    if not np.isfinite(points).all():
        raise ValueError("geometry points contain non-finite values")
    lower, upper = np.percentile(points, (0.1, 99.9), axis=0)
    extent = upper - lower
    if np.any(extent <= np.finfo(np.float64).eps):
        raise ValueError("geometry has a zero robust extent")
    return lower, upper


def normalized_points(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - lower) / (upper - lower)


def estimate_sagittal_plane(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Estimate the torso midplane without using low whisker/limb artifacts."""
    normalized = normalized_points(points, lower, upper)
    selected = (
        (normalized[:, 0] >= 0.25)
        & (normalized[:, 0] <= 0.75)
        & (normalized[:, 2] >= 0.35)
        & (normalized[:, 2] <= 0.85)
    )
    torso = np.asarray(points, dtype=np.float64)[selected]
    if len(torso) < 128:
        raise ValueError("insufficient central torso vertices for sagittal plane")
    side_low, side_high = np.percentile(torso[:, 1], (5.0, 95.0))
    plane = float(0.5 * (side_low + side_high))
    return plane, {
        "method": "central_torso_side_surface_midpoint",
        "selected_vertex_count": int(len(torso)),
        "normalized_longitudinal_range": [0.25, 0.75],
        "normalized_height_range": [0.35, 0.85],
        "side_surface_percentiles": [5.0, 95.0],
        "side_surface_values": [float(side_low), float(side_high)],
        "plane_y": plane,
    }


def _limb_corridor(
    normalized: np.ndarray,
    *,
    foot_x: float,
    attachment_x: float,
    spec: RepairSpec,
) -> np.ndarray:
    vertical = (
        (normalized[:, 2] >= 0.0)
        & (normalized[:, 2] <= spec.attachment_height_fraction)
    )
    progress = np.clip(
        normalized[:, 2] / spec.attachment_height_fraction,
        0.0,
        1.0,
    )
    center = foot_x + progress * (attachment_x - foot_x)
    half_width = (
        spec.foot_half_width_fraction
        + progress
        * (spec.attachment_half_width_fraction - spec.foot_half_width_fraction)
    )
    return vertical & (np.abs(normalized[:, 0] - center) <= half_width)


def limb_corridor_mask(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spec: RepairSpec,
) -> np.ndarray:
    """Return the two geometric limb corridors without identity exclusions."""
    spec.validate()
    normalized = normalized_points(points, lower, upper)
    front = _limb_corridor(
        normalized,
        foot_x=spec.front_foot_x_fraction,
        attachment_x=spec.front_attachment_x_fraction,
        spec=spec,
    )
    hind = _limb_corridor(
        normalized,
        foot_x=spec.hind_foot_x_fraction,
        attachment_x=spec.hind_attachment_x_fraction,
        spec=spec,
    )
    return front | hind


def _posterior_fraction(normalized: np.ndarray, spec: RepairSpec) -> np.ndarray:
    if spec.head_direction == "negative-x":
        return normalized[:, 0]
    return 1.0 - normalized[:, 0]


def identify_tail_surface(
    points: np.ndarray,
    edges: Iterable[tuple[int, int]],
    lower: np.ndarray,
    upper: np.ndarray,
    spec: RepairSpec,
    *,
    minimum_vertices: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify a conservative tail-bearing authenticated source surface.

    The old repair treated only a *high* posterior box as tail.  That admits a
    low or downward-pointing tail into the hind-leg mutation corridor.  This
    gate instead finds the surface component containing the most-posterior
    source vertex in a posterior root slab.  It may conservatively include
    adjacent rump surface; over-protection is safe and merely makes the local
    repair fail when too little hind-limb authority remains.  Height is
    deliberately absent from both seed selection and graph growth.

    Exact duplicate positions are unioned before component extraction because
    glTF UV seams may split one logical surface vertex into several raw
    vertices.  The returned mask is in the original raw-vertex domain.
    """
    spec.validate()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tail source points must be an Nx3 array")
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("tail source points are insufficient or non-finite")
    edge_list = [(int(first), int(second)) for first, second in edges]
    if any(
        first < 0
        or second < 0
        or first >= len(points)
        or second >= len(points)
        for first, second in edge_list
    ):
        raise ValueError("tail source edge index is outside the point domain")

    normalized = normalized_points(points, lower, upper)
    posterior_fraction = _posterior_fraction(normalized, spec)
    posterior = posterior_fraction >= spec.tail_protection_x_fraction
    if not np.any(posterior):
        raise ValueError("posterior tail-root slab selected no source vertices")

    # UV seams in a GLB are representation boundaries, not anatomical cuts.
    # Join only bit-identical source positions; no geometric tolerance is used.
    first_by_position: dict[bytes, int] = {}
    logical_edges = list(edge_list)
    for index in np.flatnonzero(posterior):
        key = np.asarray(points[index], dtype="<f8").tobytes()
        previous = first_by_position.setdefault(key, int(index))
        if previous != int(index):
            logical_edges.append((previous, int(index)))

    components = _components_for_selected_vertices(
        len(points),
        logical_edges,
        posterior,
    )
    if not components:
        raise ValueError("posterior tail-root slab has no source surface component")
    seed = int(np.argmax(posterior_fraction))
    containing_seed = [
        component for component in components if np.any(component == seed)
    ]
    if len(containing_seed) != 1:
        raise ValueError("most-posterior tail seed is not in exactly one component")
    tail_component = containing_seed[0]
    selected = np.zeros(len(points), dtype=bool)
    selected[tail_component] = True

    if minimum_vertices is None:
        minimum_vertices = max(32, int(round(len(points) * 0.0001)))
    chosen = points[selected]
    chosen_normalized = normalized[selected]
    chosen_posterior = posterior_fraction[selected]
    old_high_box = (
        chosen_normalized[:, 2] >= spec.tail_protection_height_fraction
    )
    posterior_span = float(np.ptp(chosen_posterior)) if len(chosen) else 0.0
    record: dict[str, Any] = {
        "method": (
            "most_posterior_authenticated_source_surface_component_"
            "height_independent"
        ),
        "height_used_for_selection": False,
        "tail_root_posterior_fraction": spec.tail_protection_x_fraction,
        "seed_vertex_index": seed,
        "seed_position": [float(value) for value in points[seed]],
        "posterior_component_count": int(len(components)),
        "posterior_component_sizes_descending": sorted(
            (int(len(component)) for component in components),
            reverse=True,
        )[:24],
        "vertex_count": int(len(chosen)),
        "minimum_vertices_required": int(minimum_vertices),
        "posterior_fraction_min": float(chosen_posterior.min(initial=1.0)),
        "posterior_fraction_max": float(chosen_posterior.max(initial=0.0)),
        "posterior_fraction_span": posterior_span,
        "minimum_posterior_span_required": 0.01,
        "vertices_below_legacy_height_cut": int(
            np.count_nonzero(~old_high_box)
        ),
        "legacy_height_cut_not_used": spec.tail_protection_height_fraction,
    }
    if len(chosen):
        record["bbox_min"] = [float(value) for value in chosen.min(axis=0)]
        record["bbox_max"] = [float(value) for value in chosen.max(axis=0)]
    record["passed"] = bool(
        len(chosen) >= minimum_vertices
        and record["posterior_fraction_max"] >= 0.95
        and posterior_span >= record["minimum_posterior_span_required"]
    )
    record["rejection_reasons"] = []
    if len(chosen) < minimum_vertices:
        record["rejection_reasons"].append("tail_surface_component_too_small")
    if record["posterior_fraction_max"] < 0.95:
        record["rejection_reasons"].append(
            "tail_surface_does_not_reach_posterior_seed_band"
        )
    if posterior_span < record["minimum_posterior_span_required"]:
        record["rejection_reasons"].append(
            "tail_surface_component_has_no_longitudinal_extent"
        )
    return selected, record


def repair_vertex_masks(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sagittal_plane_y: float,
    spec: RepairSpec,
    *,
    tail_surface: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return base-deletion and source-side donor masks.

    The masks are vertex masks rather than face masks.  Blender deletes a face
    when any of its vertices falls outside the retained domain, making the
    cut conservative before voxel union.
    """
    spec.validate()
    points = np.asarray(points, dtype=np.float64)
    normalized = normalized_points(points, lower, upper)
    front = _limb_corridor(
        normalized,
        foot_x=spec.front_foot_x_fraction,
        attachment_x=spec.front_attachment_x_fraction,
        spec=spec,
    )
    hind = _limb_corridor(
        normalized,
        foot_x=spec.hind_foot_x_fraction,
        attachment_x=spec.hind_attachment_x_fraction,
        spec=spec,
    )
    tail_protected = np.asarray(tail_surface, dtype=bool)
    if tail_protected.shape != (len(points),):
        raise ValueError("authenticated tail-surface mask does not match points")
    if not np.any(tail_protected):
        raise ValueError("authenticated tail-surface mask is empty")
    corridor = (front | hind) & ~tail_protected
    side_extent = float(upper[1] - lower[1])
    side_sign = 1.0 if spec.source_side == "positive-y" else -1.0
    signed_side_distance = (
        side_sign * (points[:, 1] - sagittal_plane_y) / side_extent
    )
    on_source_side = signed_side_distance >= spec.source_side_guard_fraction
    # At the upper shoulder/hip end, retain a narrow strip of authenticated
    # source torso around the sagittal plane.  Both the original donor and its
    # mirror can voxel-overlap this strip.  Below that height the entire repair
    # corridor is replaced, so an old whisker bar or fused far limb cannot
    # survive as a low cross-limb bridge.
    preserve_central_attachment_bridge = (
        normalized[:, 2] >= spec.central_attachment_bridge_start_fraction
    ) & (
        np.abs(signed_side_distance) < spec.source_side_guard_fraction
    )
    replace = corridor & ~preserve_central_attachment_bridge
    donor = corridor & on_source_side
    return {
        "front_corridor": front & ~tail_protected,
        "hind_corridor": hind & ~tail_protected,
        "replace_corridor": replace,
        "source_side": on_source_side,
        "donor": donor,
        "tail_protected": tail_protected,
        "preserved_central_attachment_bridge": (
            corridor & preserve_central_attachment_bridge
        ),
    }


def mask_record(
    points: np.ndarray,
    normalized: np.ndarray,
    masks: dict[str, np.ndarray],
    spec: RepairSpec,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    floor_band = normalized[:, 2] <= 0.12
    for label in (
        "front_corridor",
        "hind_corridor",
        "donor",
        "tail_protected",
        "preserved_central_attachment_bridge",
    ):
        selected = np.asarray(masks[label], dtype=bool)
        item: dict[str, Any] = {
            "vertex_count": int(np.count_nonzero(selected)),
            "floor_band_vertex_count": int(np.count_nonzero(selected & floor_band)),
        }
        if np.any(selected):
            chosen = np.asarray(points)[selected]
            item["bbox_min"] = [float(value) for value in chosen.min(axis=0)]
            item["bbox_max"] = [float(value) for value in chosen.max(axis=0)]
        result[label] = item
    front_donor = masks["front_corridor"] & masks["source_side"]
    hind_donor = masks["hind_corridor"] & masks["source_side"]
    result["front_donor_vertex_count"] = int(np.count_nonzero(front_donor))
    result["hind_donor_vertex_count"] = int(np.count_nonzero(hind_donor))
    result["front_donor_floor_band_vertex_count"] = int(
        np.count_nonzero(front_donor & floor_band)
    )
    result["hind_donor_floor_band_vertex_count"] = int(
        np.count_nonzero(hind_donor & floor_band)
    )
    result["tail_protected_donor_overlap_vertex_count"] = int(
        np.count_nonzero(masks["donor"] & masks["tail_protected"])
    )
    result["tail_protected_replacement_overlap_vertex_count"] = int(
        np.count_nonzero(masks["replace_corridor"] & masks["tail_protected"])
    )
    result["minimum_donor_vertices_required_per_limb"] = 512
    result["minimum_floor_vertices_required_per_limb"] = 64
    result["passed"] = bool(
        result["front_donor_vertex_count"] >= 512
        and result["hind_donor_vertex_count"] >= 512
        and result["front_donor_floor_band_vertex_count"] >= 64
        and result["hind_donor_floor_band_vertex_count"] >= 64
        and result["tail_protected_donor_overlap_vertex_count"] == 0
        and result["tail_protected_replacement_overlap_vertex_count"] == 0
    )
    result["source_side"] = spec.source_side
    return result


def mirror_points_across_y(points: np.ndarray, plane_y: float) -> np.ndarray:
    mirrored = np.asarray(points, dtype=np.float64).copy()
    mirrored[:, 1] = 2.0 * float(plane_y) - mirrored[:, 1]
    return mirrored


def mirror_points_across_y_with_attachment_taper(
    points: np.ndarray,
    plane_y: float,
    lower: np.ndarray,
    upper: np.ndarray,
    spec: RepairSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mirror a leg while drawing only its upper attachment into the torso."""
    spec.validate()
    points = np.asarray(points, dtype=np.float64)
    normalized_height = normalized_points(points, lower, upper)[:, 2]
    denominator = (
        spec.attachment_height_fraction
        - spec.mirrored_attachment_taper_start_fraction
    )
    progress = np.clip(
        (
            normalized_height
            - spec.mirrored_attachment_taper_start_fraction
        )
        / denominator,
        0.0,
        1.0,
    )
    lateral_scale = (
        1.0
        - progress
        * (1.0 - spec.mirrored_attachment_top_lateral_scale)
    )
    mirrored = points.copy()
    mirrored[:, 1] = (
        float(plane_y)
        - (points[:, 1] - float(plane_y)) * lateral_scale
    )
    return mirrored, {
        "method": "same_pixal_leg_mirror_with_upper_lateral_attachment_taper",
        "taper_start_height_fraction": (
            spec.mirrored_attachment_taper_start_fraction
        ),
        "attachment_top_height_fraction": spec.attachment_height_fraction,
        "top_lateral_scale": spec.mirrored_attachment_top_lateral_scale,
        "tapered_vertex_count": int(np.count_nonzero(progress > 0.0)),
        "full_strength_vertex_count": int(np.count_nonzero(progress >= 1.0)),
        "lower_limb_vertices_unchanged_except_exact_mirror": int(
            np.count_nonzero(progress == 0.0)
        ),
    }


def _float32_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def canonical_triangle_surface_signatures(
    points: np.ndarray,
    triangles: np.ndarray,
    corner_uvs: np.ndarray,
    material_indices: np.ndarray,
    *,
    selected_faces: np.ndarray | None = None,
) -> Counter[str]:
    """Return winding-preserving surface signatures independent of index IDs.

    Blender/glTF may split one logical vertex at a UV seam or renumber vertices
    on export.  The bounded identity contract is therefore expressed per
    triangle corner as exact float32 ``POSITION`` + ``TEXCOORD_0`` together
    with the material assignment.  Cyclic index rotation is ignored, while
    winding reversal is not.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    corner_uvs = np.asarray(corner_uvs, dtype=np.float64)
    material_indices = np.asarray(material_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface points must be an Nx3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("surface triangles must be an Fx3 array")
    if corner_uvs.shape != (len(triangles), 3, 2):
        raise ValueError("surface corner UVs must have shape (F, 3, 2)")
    if material_indices.shape != (len(triangles),):
        raise ValueError("surface material indices must have shape (F,)")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("surface triangle index is outside point domain")
    if selected_faces is None:
        selected_faces = np.ones(len(triangles), dtype=bool)
    else:
        selected_faces = np.asarray(selected_faces, dtype=bool)
        if selected_faces.shape != (len(triangles),):
            raise ValueError("selected surface faces must have shape (F,)")

    result: Counter[str] = Counter()
    for face_index in np.flatnonzero(selected_faces):
        corners = []
        for loop_index in range(3):
            vertex_index = triangles[face_index, loop_index]
            corners.append(
                _float32_bytes(points[vertex_index])
                + _float32_bytes(corner_uvs[face_index, loop_index])
            )
        rotations = (
            b"".join(corners),
            b"".join(corners[1:] + corners[:1]),
            b"".join(corners[2:] + corners[:2]),
        )
        payload = (
            struct.pack("<q", int(material_indices[face_index]))
            + min(rotations)
        )
        result[hashlib.sha256(payload).hexdigest()] += 1
    return result


def audit_expected_surface_signatures(
    expected: Counter[str],
    actual: Counter[str],
) -> dict[str, Any]:
    """Require exact multiplicity for every immutable source triangle."""
    missing = expected - actual
    unexpected_duplicates = Counter(
        {
            signature: actual[signature] - count
            for signature, count in expected.items()
            if actual[signature] > count
        }
    )

    def digest(counter: Counter[str]) -> str:
        hasher = hashlib.sha256()
        for signature, count in sorted(counter.items()):
            hasher.update(signature.encode("ascii"))
            hasher.update(struct.pack("<Q", int(count)))
        return hasher.hexdigest()

    return {
        "method": (
            "exact_float32_position_uv_material_triangle_multiset_"
            "with_winding_preserved"
        ),
        "expected_triangle_count": int(sum(expected.values())),
        "actual_matching_triangle_count": int(
            sum(min(count, actual[signature]) for signature, count in expected.items())
        ),
        "missing_triangle_count": int(sum(missing.values())),
        "unexpected_duplicate_triangle_count": int(
            sum(unexpected_duplicates.values())
        ),
        "expected_signature_sha256": digest(expected),
        "missing_signature_sha256": digest(missing),
        "unexpected_duplicate_signature_sha256": digest(
            unexpected_duplicates
        ),
        "passed": not missing and not unexpected_duplicates,
    }


def audit_immutable_source_topology(
    points: np.ndarray,
    triangles: np.ndarray,
    mutable_vertices: np.ndarray,
) -> dict[str, Any]:
    """Reject a source whose non-manifold repair would escape the corridors.

    Exact-position logical vertices collapse only glTF representation splits;
    no tolerance weld is used.  Every logical edge not wholly inside the
    mutable vertex domain must already have exactly two oppositely-oriented
    incident triangles.  Otherwise producing a closed output necessarily
    changes topology outside the authorized corridors.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    mutable_vertices = np.asarray(mutable_vertices, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("topology points must be an Nx3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("topology triangles must be an Fx3 array")
    if mutable_vertices.shape != (len(points),):
        raise ValueError("mutable vertex mask must have shape (N,)")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("topology triangle index is outside point domain")

    logical_by_key: dict[bytes, int] = {}
    logical_ids = np.empty(len(points), dtype=np.int64)
    logical_mutable: list[bool] = []
    for index, point in enumerate(points):
        key = _float32_bytes(point)
        logical = logical_by_key.get(key)
        if logical is None:
            logical = len(logical_mutable)
            logical_by_key[key] = logical
            logical_mutable.append(bool(mutable_vertices[index]))
        else:
            # A logical source position is mutable only when every raw split
            # is mutable.  This prevents a UV-seam duplicate from broadening
            # the repair authority.
            logical_mutable[logical] = (
                logical_mutable[logical] and bool(mutable_vertices[index])
            )
        logical_ids[index] = logical

    occurrences: dict[tuple[int, int], list[int]] = {}
    adjacency: list[set[int]] = [set() for _ in logical_mutable]
    used_logical_vertices: set[int] = set()
    degenerate_triangle_count = 0
    for triangle in triangles:
        logical = [int(logical_ids[index]) for index in triangle]
        if len(set(logical)) != 3:
            degenerate_triangle_count += 1
            continue
        used_logical_vertices.update(logical)
        for first, second in zip(logical, logical[1:] + logical[:1]):
            key = (min(first, second), max(first, second))
            direction = 1 if (first, second) == key else -1
            occurrences.setdefault(key, []).append(direction)
            adjacency[first].add(second)
            adjacency[second].add(first)

    outside_boundary = 0
    outside_over_two = 0
    outside_orientation = 0
    mutable_defects = 0
    for (first, second), directions in occurrences.items():
        mutable = logical_mutable[first] and logical_mutable[second]
        valid = len(directions) == 2 and sum(directions) == 0
        if valid:
            continue
        if mutable:
            mutable_defects += 1
        elif len(directions) == 1:
            outside_boundary += 1
        elif len(directions) > 2:
            outside_over_two += 1
        else:
            outside_orientation += 1
    unseen = set(used_logical_vertices)
    component_records = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        component_records.append(
            {
                "size": int(len(component)),
                "wholly_mutable": bool(
                    all(logical_mutable[index] for index in component)
                ),
            }
        )
    component_records.sort(key=lambda item: item["size"], reverse=True)
    immutable_component_count = sum(
        not item["wholly_mutable"] for item in component_records
    )
    passed = bool(
        outside_boundary == 0
        and outside_over_two == 0
        and outside_orientation == 0
        and degenerate_triangle_count == 0
        and immutable_component_count == 1
    )
    return {
        "method": (
            "exact_position_logical_edge_incidence_outside_authorized_corridors"
        ),
        "raw_vertex_count": int(len(points)),
        "logical_position_count": int(len(logical_mutable)),
        "triangle_count": int(len(triangles)),
        "logical_edge_count": int(len(occurrences)),
        "logical_component_count": int(len(component_records)),
        "immutable_component_count": int(immutable_component_count),
        "wholly_mutable_component_count": int(
            len(component_records) - immutable_component_count
        ),
        "logical_component_sizes_descending": [
            int(item["size"]) for item in component_records[:24]
        ],
        "outside_corridor_boundary_edges": int(outside_boundary),
        "outside_corridor_edges_over_two_faces": int(outside_over_two),
        "outside_corridor_orientation_mismatch_edges": int(
            outside_orientation
        ),
        "inside_corridor_defective_edges_available_for_local_repair": int(
            mutable_defects
        ),
        "degenerate_triangle_count": int(degenerate_triangle_count),
        "passed": passed,
        "failure_meaning": (
            None
            if passed
            else (
                "a watertight output would require changing topology outside "
                "the authorized limb corridors"
            )
        ),
    }


def _components_for_selected_vertices(
    point_count: int,
    edges: Iterable[tuple[int, int]],
    selected: np.ndarray,
) -> list[np.ndarray]:
    parent = np.arange(point_count, dtype=np.int64)
    rank = np.zeros(point_count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
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
        first = int(first)
        second = int(second)
        if selected[first] and selected[second]:
            union(first, second)
    grouped: dict[int, list[int]] = {}
    for vertex in np.flatnonzero(selected):
        grouped.setdefault(find(int(vertex)), []).append(int(vertex))
    return [
        np.asarray(indices, dtype=np.int64) for indices in grouped.values()
    ]


def audit_low_slice_limb_chains(
    points: np.ndarray,
    edges: Iterable[tuple[int, int]],
    sagittal_plane_y: float,
    spec: RepairSpec,
    *,
    minimum_component_vertices: int | None = None,
) -> dict[str, Any]:
    """Require four floor-to-slice components and reject low membranes."""
    spec.validate()
    points = np.asarray(points, dtype=np.float64)
    lower, upper = robust_bounds(points)
    extent = upper - lower
    threshold = float(
        lower[2] + spec.low_slice_height_fraction * extent[2]
    )
    selected = points[:, 2] <= threshold
    edge_list = [(int(first), int(second)) for first, second in edges]
    # Treat bit-identical glTF UV-seam splits as one logical surface vertex.
    # This does not tolerate or invent a weld between distinct positions.
    first_by_position: dict[bytes, int] = {}
    for index in np.flatnonzero(selected):
        key = np.asarray(points[index], dtype="<f8").tobytes()
        previous = first_by_position.setdefault(key, int(index))
        if previous != int(index):
            edge_list.append((previous, int(index)))
    components = _components_for_selected_vertices(
        len(points),
        edge_list,
        selected,
    )
    if minimum_component_vertices is None:
        minimum_component_vertices = max(32, int(round(len(points) * 0.002)))
    substantial = [
        component
        for component in components
        if len(component) >= minimum_component_vertices
    ]
    substantial.sort(key=len, reverse=True)
    x_split = 0.5 * (
        spec.front_attachment_x_fraction + spec.hind_attachment_x_fraction
    )
    records = []
    quadrant_counts = {
        "front_negative_y": 0,
        "front_positive_y": 0,
        "hind_negative_y": 0,
        "hind_positive_y": 0,
    }
    floor_limit = float(lower[2] + 0.08 * extent[2])
    chain_top_limit = float(threshold - 0.06 * extent[2])
    for component in substantial:
        chosen = points[component]
        centroid = chosen.mean(axis=0)
        x_fraction = float((centroid[0] - lower[0]) / extent[0])
        longitudinal = "front" if (
            x_fraction < x_split
            if spec.head_direction == "negative-x"
            else x_fraction > x_split
        ) else "hind"
        side = "positive_y" if centroid[1] >= sagittal_plane_y else "negative_y"
        quadrant = f"{longitudinal}_{side}"
        quadrant_counts[quadrant] += 1
        records.append(
            {
                "vertex_count": int(len(component)),
                "centroid": [float(value) for value in centroid],
                "bbox_min": [float(value) for value in chosen.min(axis=0)],
                "bbox_max": [float(value) for value in chosen.max(axis=0)],
                "quadrant": quadrant,
                "touches_floor_band": bool(chosen[:, 2].min() <= floor_limit),
                "reaches_slice_top_band": bool(
                    chosen[:, 2].max() >= chain_top_limit
                ),
            }
        )
    top_four = records[:4]
    four_components = len(substantial) >= 4
    unique_quadrants = bool(
        len(top_four) == 4
        and all(quadrant_counts[label] == 1 for label in quadrant_counts)
    )
    four_contacts = bool(
        len(top_four) == 4
        and all(record["touches_floor_band"] for record in top_four)
    )
    four_vertical_chains = bool(
        len(top_four) == 4
        and all(record["reaches_slice_top_band"] for record in top_four)
    )
    fifth_size = len(substantial[4]) if len(substantial) > 4 else 0
    fifth_is_small = bool(
        len(substantial) <= 4
        or fifth_size < 0.4 * len(substantial[3])
    )
    passed = bool(
        four_components
        and unique_quadrants
        and four_contacts
        and four_vertical_chains
        and fifth_is_small
    )
    reasons = []
    if not four_components:
        reasons.append("low_slice_has_fewer_than_four_substantial_components")
    if four_components and not unique_quadrants:
        reasons.append("low_slice_components_do_not_map_one_to_each_limb_quadrant")
    if not four_contacts:
        reasons.append("not_all_four_limb_components_reach_floor_band")
    if not four_vertical_chains:
        reasons.append("not_all_four_limb_components_reach_attachment_slice")
    if not fifth_is_small:
        reasons.append("ambiguous_fifth_low_slice_component")
    return {
        "method": "four_disconnected_floor_to_attachment_induced_components",
        "blender_up_axis": "positive-z",
        "height_fraction": spec.low_slice_height_fraction,
        "slice_threshold": threshold,
        "robust_floor": float(lower[2]),
        "robust_ceiling": float(upper[2]),
        "minimum_component_vertices": int(minimum_component_vertices),
        "substantial_component_sizes_descending": [
            int(len(component)) for component in substantial[:12]
        ],
        "quadrant_counts": quadrant_counts,
        "components": records[:12],
        "checks": {
            "four_substantial_low_components": four_components,
            "one_component_per_limb_quadrant": unique_quadrants,
            "four_floor_contacts": four_contacts,
            "four_floor_to_attachment_chains": four_vertical_chains,
            "no_ambiguous_fifth_component": fifth_is_small,
            "no_low_cross_limb_membrane": unique_quadrants,
        },
        "rejection_reasons": reasons,
        "passed": passed,
    }
