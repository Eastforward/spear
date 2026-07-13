#!/usr/bin/env python3
"""Audit image-to-3D quadruped geometry before LOD or skin transfer.

The audit deliberately does not infer the animal's head direction or apply a
yaw correction.  A rigid global yaw is recorded as orientation evidence but is
not a defect: only curvature remaining after that rigid axis is removed is used
as a torso-shape gate.  The other gates measure degenerate triangles and edges
shared by more than two faces.  glTF's UV/normal vertex copies are
position-indexed for the topology audit so ordinary texture seams are not
reported as holes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh


SCHEMA = "avengine_quadruped_i23d_geometry_audit_v2"
BEND_PASS_DEG = 5.0
BEND_REVIEW_DEG = 10.0
NONMANIFOLD_PASS_RATIO = 1.0e-4
NONMANIFOLD_REJECT_RATIO = 1.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_meshes(scene: trimesh.Scene) -> Iterable[trimesh.Trimesh]:
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        yield mesh


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(_scene_meshes(loaded))
        if not meshes:
            raise ValueError(f"mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(meshes)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"invalid mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"triangular faces are required: {path}")
    return vertices, faces


def position_indexed_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Count physical topology after exact glTF seam copies share an index."""
    unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    indexed_faces = inverse[faces]
    nondegenerate = (
        (indexed_faces[:, 0] != indexed_faces[:, 1])
        & (indexed_faces[:, 1] != indexed_faces[:, 2])
        & (indexed_faces[:, 2] != indexed_faces[:, 0])
    )
    valid_faces = indexed_faces[nondegenerate]
    edges = np.concatenate(
        (
            valid_faces[:, [0, 1]],
            valid_faces[:, [1, 2]],
            valid_faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    face_count = int(len(valid_faces))
    nonmanifold = int(np.count_nonzero(multiplicity > 2))
    return {
        "imported_vertices": int(len(vertices)),
        "position_unique_vertices": int(len(unique_vertices)),
        "imported_triangles": int(len(faces)),
        "position_indexed_triangles": face_count,
        "degenerate_triangles_after_position_indexing": int(
            len(faces) - face_count
        ),
        "boundary_edges": int(np.count_nonzero(multiplicity == 1)),
        "manifold_two_face_edges": int(np.count_nonzero(multiplicity == 2)),
        "nonmanifold_edges_over_two_faces": nonmanifold,
        "maximum_edge_face_multiplicity": int(multiplicity.max(initial=0)),
        "nonmanifold_edge_ratio_per_triangle": (
            float(nonmanifold / face_count) if face_count else 1.0
        ),
    }


def torso_midline_yaw(
    vertices: np.ndarray,
    *,
    central_percentiles: tuple[float, float] = (25.0, 75.0),
    bins: int = 24,
) -> dict[str, Any]:
    """Fit the top-view torso surface midline in glTF +X/+Y-up/+Z-side.

    Head, tail and most leg vertices are excluded.  Within each longitudinal
    bin, the midpoint between robust side-surface percentiles estimates the
    torso center independently of unequal sampling density on either flank.
    """
    points = np.unique(vertices, axis=0)
    x_start, x_end = np.percentile(points[:, 0], central_percentiles)
    up_low, up_high = np.percentile(points[:, 1], (2.0, 98.0))
    torso_floor = up_low + 0.35 * (up_high - up_low)
    selected = points[
        (points[:, 0] >= x_start)
        & (points[:, 0] <= x_end)
        & (points[:, 1] >= torso_floor)
    ]
    edges = np.linspace(x_start, x_end, bins + 1)
    centers = []
    for start, end in zip(edges[:-1], edges[1:]):
        section = selected[
            (selected[:, 0] >= start) & (selected[:, 0] < end)
        ]
        if len(section) < 30:
            continue
        side_midpoint = 0.5 * (
            np.percentile(section[:, 2], 5.0)
            + np.percentile(section[:, 2], 95.0)
        )
        centers.append((0.5 * (start + end), side_midpoint))
    if len(centers) < max(8, bins // 2):
        raise ValueError("insufficient torso sections for a stable midline fit")
    center_array = np.asarray(centers, dtype=np.float64)
    slope, intercept = np.polyfit(center_array[:, 0], center_array[:, 1], 1)
    predicted = slope * center_array[:, 0] + intercept
    residual = center_array[:, 1] - predicted
    total = center_array[:, 1] - center_array[:, 1].mean()
    denominator = float(total @ total)
    r_squared = 1.0 - float(residual @ residual) / denominator if denominator else 1.0
    # A straight animal can have any rigid horizontal yaw.  Rotate the sampled
    # centerline into its unsigned PCA axis before measuring shape.  This keeps
    # a straight +17 degree mesh at zero bend while still exposing an arc or an
    # S-shaped torso.  PCA deliberately does not decide which endpoint is the
    # head, so cardinal direction remains a separate human/review concern.
    centered = center_array - center_array.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[0] < 0.0:
        axis = -axis
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    longitudinal = centered @ axis
    lateral = centered @ normal
    order = np.argsort(longitudinal)
    longitudinal = longitudinal[order]
    lateral = lateral[order]
    longitudinal_span = float(np.ptp(longitudinal))
    if longitudinal_span <= np.finfo(np.float64).eps:
        raise ValueError("torso centerline has no stable longitudinal span")
    half_span = 0.5 * longitudinal_span
    normalized_longitudinal = longitudinal / half_span
    polynomial_degree = min(3, len(center_array) - 1)
    curve = np.polyfit(normalized_longitudinal, lateral, polynomial_degree)
    tangent_lateral_per_forward = np.polyval(
        np.polyder(curve), normalized_longitudinal
    ) / half_span
    local_deviation_degrees = np.degrees(
        np.arctan(tangent_lateral_per_forward)
    )
    bend_p95_degrees = float(
        np.percentile(np.abs(local_deviation_degrees), 95.0)
    )
    bend_max_degrees = float(np.max(np.abs(local_deviation_degrees)))
    pca_global_axis_yaw_degrees = float(
        math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    )
    lateral_rms_ratio = float(
        np.sqrt(np.mean(np.square(lateral))) / longitudinal_span
    )
    lateral_peak_ratio = float(np.max(np.abs(lateral)) / longitudinal_span)
    return {
        "coordinate_frame": "gltf_positive_x_forward_positive_y_up_positive_z_side",
        "central_longitudinal_percentiles": list(central_percentiles),
        "torso_floor_fraction_of_robust_height": 0.35,
        "surface_side_percentiles": [5.0, 95.0],
        "section_count": int(len(centers)),
        "selected_vertex_count": int(len(selected)),
        "side_slope_per_forward_unit": float(slope),
        "yaw_degrees": float(math.degrees(math.atan(float(slope)))),
        "global_axis_yaw_degrees": pca_global_axis_yaw_degrees,
        "global_axis_semantics": (
            "rigid_unsigned_orientation_evidence_only_not_a_shape_defect"
        ),
        "centerline_bend_p95_degrees": bend_p95_degrees,
        "centerline_bend_max_degrees": bend_max_degrees,
        "centerline_lateral_rms_ratio": lateral_rms_ratio,
        "centerline_lateral_peak_ratio": lateral_peak_ratio,
        "centerline_curve_degree": polynomial_degree,
        "centerline_shape_semantics": (
            "local_tangent_deviation_after_removing_rigid_pca_axis"
        ),
        "fit_r_squared": float(r_squared),
    }


def decision(topology: dict[str, Any], midline: dict[str, Any]) -> dict[str, Any]:
    bend = abs(float(midline["centerline_bend_p95_degrees"]))
    ratio = float(topology["nonmanifold_edge_ratio_per_triangle"])
    rejected = []
    review = []
    if bend > BEND_REVIEW_DEG:
        rejected.append("torso_centerline_bend_exceeds_10_degrees")
    elif bend > BEND_PASS_DEG:
        review.append("torso_centerline_bend_between_5_and_10_degrees")
    if ratio > NONMANIFOLD_REJECT_RATIO:
        rejected.append("nonmanifold_edge_ratio_exceeds_0_001")
    elif ratio > NONMANIFOLD_PASS_RATIO:
        review.append("nonmanifold_edge_ratio_between_0_0001_and_0_001")
    if rejected:
        status = "reject_before_lod_and_binding"
    elif review:
        status = "manual_source_geometry_review_required"
    else:
        status = "passed_automatic_geometry_measurements"
    return {
        "status": status,
        "rejection_reasons": rejected,
        "manual_review_reasons": review,
        "cardinal_orientation_inference": "disabled",
        "global_yaw_is_a_rejection_criterion": False,
        "thresholds": {
            "torso_centerline_bend_pass_max_degrees": BEND_PASS_DEG,
            "torso_centerline_bend_reject_above_degrees": BEND_REVIEW_DEG,
            "nonmanifold_pass_max_ratio": NONMANIFOLD_PASS_RATIO,
            "nonmanifold_reject_above_ratio": NONMANIFOLD_REJECT_RATIO,
        },
    }


def audit(path: Path, label: str) -> dict[str, Any]:
    vertices, faces = load_mesh(path)
    topology = position_indexed_topology(vertices, faces)
    fits = [
        torso_midline_yaw(vertices, central_percentiles=span)
        for span in ((25.0, 75.0), (30.0, 70.0), (35.0, 65.0))
    ]
    midline = {
        "primary": fits[0],
        "sensitivity_yaw_degrees": [fit["yaw_degrees"] for fit in fits],
        "sensitivity_global_axis_yaw_degrees": [
            fit["global_axis_yaw_degrees"] for fit in fits
        ],
        "sensitivity_centerline_bend_p95_degrees": [
            fit["centerline_bend_p95_degrees"] for fit in fits
        ],
        "sensitivity_central_percentiles": [
            fit["central_longitudinal_percentiles"] for fit in fits
        ],
    }
    return {
        "label": label,
        "mesh": {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        },
        "topology": topology,
        "torso_midline": midline,
        "decision": decision(topology, fits[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each comparison mesh.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    labels = set()
    for item in args.mesh:
        if "=" not in item:
            raise SystemExit("--mesh must use LABEL=PATH")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not label or label in labels or not path.is_file() or path.is_symlink():
            raise SystemExit(f"invalid or duplicate mesh record: {item}")
        labels.add(label)
        records.append(audit(path, label))
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "prebind_geometry_measurement_without_direction_inference",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"QUADRUPED_I23D_GEOMETRY_AUDIT_OK {args.output}")


if __name__ == "__main__":
    main()
