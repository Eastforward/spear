"""Measure animated mesh deformation without changing or exporting the asset.

The gate is deliberately independent of animal taxonomy and bone naming.  It
compares every sampled Walk/Idle pose against the authored rest-pose topology.
Large edge extension and triangle-area growth expose the visible fans and
membranes produced by cross-limb weights even when the GLB is technically
readable and its face count is unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np


SCHEMA = "avengine_skinned_deformation_audit_v1"


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--samples", type=int, default=24)
    # The authored Quaternius Cat/Dog/Wolf/Cow walking controls reach 0.0624
    # of the rest-pose diagonal at a handful of joints without producing a
    # visible membrane. Keep the review threshold above that measured control
    # while retaining the 0.08 hard reject used by Pixal cross-limb failures.
    parser.add_argument("--review-edge-extension-ratio", type=float, default=0.07)
    parser.add_argument("--reject-edge-extension-ratio", type=float, default=0.08)
    parser.add_argument("--review-edge-stretch-ratio", type=float, default=2.0)
    parser.add_argument("--reject-edge-stretch-ratio", type=float, default=4.0)
    parser.add_argument("--review-area-stretch-ratio", type=float, default=3.0)
    parser.add_argument("--reject-area-stretch-ratio", type=float, default=8.0)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, values_at=(50.0, 95.0, 99.0, 99.9)):
    if len(values) == 0:
        return {str(value): 0.0 for value in values_at}
    return {
        str(value): float(np.percentile(values, value)) for value in values_at
    }


def evaluated_geometry(mesh_object, depsgraph):
    evaluated = mesh_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float64)))
        vertices = (homogeneous @ matrix.T)[:, :3]
        edges = np.asarray(
            [edge.vertices[:] for edge in mesh.edges], dtype=np.int64
        )
        triangles = []
        for polygon in mesh.polygons:
            indices = list(polygon.vertices)
            for index in range(1, len(indices) - 1):
                triangles.append((indices[0], indices[index], indices[index + 1]))
        faces = np.asarray(triangles, dtype=np.int64)
    finally:
        evaluated.to_mesh_clear()
    return vertices, edges, faces


def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    second = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(first, second), axis=1)


def symmetric_ratio(current: np.ndarray, rest: np.ndarray) -> np.ndarray:
    return np.maximum(current / rest, rest / current)


def frame_metrics(
    vertices,
    edges,
    faces,
    *,
    rest_edge_lengths,
    rest_areas,
    rest_diagonal,
    edge_mask,
    area_mask,
):
    current_edges = edge_lengths(vertices, edges)
    current_areas = triangle_areas(vertices, faces)
    edge_stretch_ratio = symmetric_ratio(
        current_edges[edge_mask], rest_edge_lengths[edge_mask]
    )
    edge_extension_ratio_of_rest_diagonal = np.maximum(
        current_edges[edge_mask] - rest_edge_lengths[edge_mask], 0.0
    ) / rest_diagonal
    triangle_area_stretch_ratio = symmetric_ratio(
        current_areas[area_mask], rest_areas[area_mask]
    )
    return {
        "edge_stretch_ratio": {
            "maximum": float(edge_stretch_ratio.max(initial=1.0)),
            "percentiles": percentile(edge_stretch_ratio),
            "count_over_2": int(np.count_nonzero(edge_stretch_ratio > 2.0)),
            "count_over_4": int(np.count_nonzero(edge_stretch_ratio > 4.0)),
        },
        "edge_extension_ratio_of_rest_diagonal": {
            "maximum": float(
                edge_extension_ratio_of_rest_diagonal.max(initial=0.0)
            ),
            "percentiles": percentile(edge_extension_ratio_of_rest_diagonal),
            "count_over_0_04": int(
                np.count_nonzero(edge_extension_ratio_of_rest_diagonal > 0.04)
            ),
            "count_over_0_08": int(
                np.count_nonzero(edge_extension_ratio_of_rest_diagonal > 0.08)
            ),
        },
        "triangle_area_stretch_ratio": {
            "maximum": float(triangle_area_stretch_ratio.max(initial=1.0)),
            "percentiles": percentile(triangle_area_stretch_ratio),
            "count_over_3": int(np.count_nonzero(triangle_area_stretch_ratio > 3.0)),
            "count_over_8": int(np.count_nonzero(triangle_area_stretch_ratio > 8.0)),
        },
        "surface_area_ratio_to_rest": float(
            current_areas.sum() / max(rest_areas.sum(), 1.0e-12)
        ),
    }


def worst_over_frames(frames):
    return {
        "maximum_edge_stretch_ratio": max(
            frame["metrics"]["edge_stretch_ratio"]["maximum"] for frame in frames
        ),
        "maximum_edge_extension_ratio_of_rest_diagonal": max(
            frame["metrics"]["edge_extension_ratio_of_rest_diagonal"]["maximum"]
            for frame in frames
        ),
        "maximum_triangle_area_stretch_ratio": max(
            frame["metrics"]["triangle_area_stretch_ratio"]["maximum"]
            for frame in frames
        ),
        "maximum_surface_area_ratio_to_rest": max(
            frame["metrics"]["surface_area_ratio_to_rest"] for frame in frames
        ),
    }


def action_decision(worst, thresholds):
    extension = worst["maximum_edge_extension_ratio_of_rest_diagonal"]
    edge = worst["maximum_edge_stretch_ratio"]
    area = worst["maximum_triangle_area_stretch_ratio"]
    reject = (
        extension > thresholds["reject_edge_extension_ratio"]
        and (
            edge > thresholds["reject_edge_stretch_ratio"]
            or area > thresholds["reject_area_stretch_ratio"]
        )
    )
    review = (
        extension > thresholds["review_edge_extension_ratio"]
        and (
            edge > thresholds["review_edge_stretch_ratio"]
            or area > thresholds["review_area_stretch_ratio"]
        )
    )
    if reject:
        return "reject_visible_skinning_fan_or_membrane"
    if review:
        return "manual_review_local_deformation"
    return "passed_automatic_deformation_measurements"


def main():
    args = parse_argv()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"missing or unsafe input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 4 <= args.samples <= 120:
        raise SystemExit("--samples must be in [4, 120]")
    thresholds = {
        "review_edge_extension_ratio": args.review_edge_extension_ratio,
        "reject_edge_extension_ratio": args.reject_edge_extension_ratio,
        "review_edge_stretch_ratio": args.review_edge_stretch_ratio,
        "reject_edge_stretch_ratio": args.reject_edge_stretch_ratio,
        "review_area_stretch_ratio": args.review_area_stretch_ratio,
        "reject_area_stretch_ratio": args.reject_area_stretch_ratio,
    }

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if not meshes or len(armatures) != 1:
        raise SystemExit("input must contain a mesh and exactly one armature")
    body = max(meshes, key=lambda item: len(item.data.vertices))
    armature = armatures[0]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    armature.animation_data_clear()
    armature.data.pose_position = "REST"
    scene.frame_set(1)
    bpy.context.view_layer.update()
    rest_vertices, edges, faces = evaluated_geometry(body, depsgraph)
    rest_extent = np.ptp(rest_vertices, axis=0)
    rest_diagonal = float(np.linalg.norm(rest_extent))
    rest_edge_lengths = edge_lengths(rest_vertices, edges)
    rest_areas = triangle_areas(rest_vertices, faces)
    edge_mask = rest_edge_lengths > rest_diagonal * 1.0e-5
    area_mask = rest_areas > rest_diagonal * rest_diagonal * 1.0e-10

    requested = args.action or ["Walking", "Idle"]
    available = list(bpy.data.actions)
    records = []
    for requested_name in requested:
        action = next(
            (
                candidate
                for candidate in available
                if requested_name.lower() in candidate.name.lower()
            ),
            None,
        )
        if action is None:
            raise SystemExit(
                f"missing action {requested_name}; available={[item.name for item in available]}"
            )
        armature.data.pose_position = "POSE"
        armature.animation_data_create()
        armature.animation_data.action = action
        start, end = action.frame_range
        sampled_frames = np.linspace(start, end, args.samples)
        frames = []
        for frame in sampled_frames:
            scene.frame_set(int(round(float(frame))))
            bpy.context.view_layer.update()
            vertices, current_edges, current_faces = evaluated_geometry(body, depsgraph)
            if (
                len(vertices) != len(rest_vertices)
                or not np.array_equal(current_edges, edges)
                or not np.array_equal(current_faces, faces)
            ):
                raise SystemExit("evaluated topology changed across animation frames")
            frames.append(
                {
                    "source_frame": float(frame),
                    "evaluated_frame": int(round(float(frame))),
                    "metrics": frame_metrics(
                        vertices,
                        edges,
                        faces,
                        rest_edge_lengths=rest_edge_lengths,
                        rest_areas=rest_areas,
                        rest_diagonal=rest_diagonal,
                        edge_mask=edge_mask,
                        area_mask=area_mask,
                    ),
                }
            )
        worst = worst_over_frames(frames)
        records.append(
            {
                "requested_action": requested_name,
                "resolved_action": action.name,
                "frame_range": [float(start), float(end)],
                "sampled_frames": frames,
                "worst_case": worst,
                "decision": action_decision(worst, thresholds),
            }
        )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "input_sha256": sha256_file(source),
        "input_size_bytes": source.stat().st_size,
        "body_object": body.name,
        "armature_object": armature.name,
        "rest_geometry": {
            "vertices": int(len(rest_vertices)),
            "edges": int(len(edges)),
            "triangles": int(len(faces)),
            "diagonal": rest_diagonal,
            "measured_edges": int(np.count_nonzero(edge_mask)),
            "measured_triangles": int(np.count_nonzero(area_mask)),
        },
        "thresholds": thresholds,
        "actions": records,
        "overall": (
            "rejected"
            if any(record["decision"].startswith("reject_") for record in records)
            else "manual_review_required"
            if any(record["decision"].startswith("manual_") for record in records)
            else "passed"
        ),
        "formal_dataset_registration_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"SKINNED_DEFORMATION_AUDIT_OK overall={payload['overall']} output={output}")


if __name__ == "__main__":
    main()
