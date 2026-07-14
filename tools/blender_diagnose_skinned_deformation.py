"""Trace visible skinning stretch back to topology and vertex weights.

This diagnostic never edits or exports the input.  It complements
``blender_audit_skinned_deformation.py`` by recording the exact animated edges
that extend the most, their rest/posed coordinates, and the bone influences at
both endpoints.  For one-root quadrupeds it also labels each influence with a
geometry-inferred semantic chain, which exposes cross-leg bridge edges and
vertices whose weights mix incompatible limbs.
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
from tools.generated_quadruped_semantics import (  # noqa: E402
    infer_quadruped_semantics,
)


SCHEMA = "avengine_skinned_deformation_diagnostic_v1"
LIMB_CHAINS = {
    "front_side_negative",
    "front_side_positive",
    "hind_side_negative",
    "hind_side_positive",
}


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", default="Walking")
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument("--minimum-influence", type=float, default=0.01)
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


def require_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def bone_records(armature):
    records = []
    for bone in armature.data.bones:
        records.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "children": sorted(child.name for child in bone.children),
                "head_world": list(armature.matrix_world @ bone.head_local),
                "tail_world": list(armature.matrix_world @ bone.tail_local),
            }
        )
    return records


def semantic_labels(armature, rest_vertices, front_axis):
    minimum = rest_vertices.min(axis=0)
    extent = np.ptp(rest_vertices, axis=0)
    semantics = infer_quadruped_semantics(
        bone_records(armature),
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=front_axis,
    )
    labels = {}
    for label, chain in semantics.chains().items():
        for name in chain:
            if name in labels:
                raise RuntimeError(f"bone appears in multiple semantic chains: {name}")
            labels[name] = label
    if set(labels) != set(armature.data.bones.keys()):
        raise RuntimeError("semantic labels do not cover the armature")
    return semantics, labels


def vertex_influences(body, bone_labels, minimum):
    group_names = {group.index: group.name for group in body.vertex_groups}
    result = []
    for vertex in body.data.vertices:
        values = []
        for membership in vertex.groups:
            name = group_names.get(membership.group)
            if name not in bone_labels or membership.weight < minimum:
                continue
            values.append(
                {
                    "bone": name,
                    "semantic_chain": bone_labels[name],
                    "weight": float(membership.weight),
                }
            )
        values.sort(key=lambda item: (-item["weight"], item["bone"]))
        result.append(values)
    return result


def dominant_chain(influences):
    return influences[0]["semantic_chain"] if influences else "unweighted"


def mixed_limb_chains(influences):
    return sorted(
        {
            item["semantic_chain"]
            for item in influences
            if item["semantic_chain"] in LIMB_CHAINS
        }
    )


def edge_static_classification(edges, influences):
    dominant = [dominant_chain(value) for value in influences]
    cross_limb = []
    for edge_index, (first, second) in enumerate(edges):
        left = dominant[int(first)]
        right = dominant[int(second)]
        if left in LIMB_CHAINS and right in LIMB_CHAINS and left != right:
            cross_limb.append(edge_index)
    mixed_vertices = [
        index
        for index, value in enumerate(influences)
        if len(mixed_limb_chains(value)) > 1
    ]
    return dominant, np.asarray(cross_limb, dtype=np.int64), mixed_vertices


def serializable_point(value):
    return [float(component) for component in value]


def edge_record(
    *,
    edge_index,
    frame,
    rest_vertices,
    posed_vertices,
    edges,
    rest_lengths,
    posed_lengths,
    rest_diagonal,
    influences,
    dominant,
):
    first, second = (int(value) for value in edges[edge_index])
    left = dominant[first]
    right = dominant[second]
    return {
        "edge_index": int(edge_index),
        "evaluated_frame": int(frame),
        "vertices": [first, second],
        "rest_length": float(rest_lengths[edge_index]),
        "posed_length": float(posed_lengths[edge_index]),
        "positive_extension_ratio_of_rest_diagonal": float(
            max(posed_lengths[edge_index] - rest_lengths[edge_index], 0.0)
            / rest_diagonal
        ),
        "symmetric_stretch_ratio": float(
            max(
                posed_lengths[edge_index] / rest_lengths[edge_index],
                rest_lengths[edge_index] / posed_lengths[edge_index],
            )
        ),
        "endpoint_semantics": [left, right],
        "cross_limb_dominant_edge": bool(
            left in LIMB_CHAINS and right in LIMB_CHAINS and left != right
        ),
        "endpoints": [
            {
                "vertex_index": index,
                "rest_world": serializable_point(rest_vertices[index]),
                "posed_world": serializable_point(posed_vertices[index]),
                "influences": influences[index],
                "mixed_limb_chains": mixed_limb_chains(influences[index]),
            }
            for index in (first, second)
        ],
    }


def main():
    args = parse_argv()
    source = require_input(args.input)
    output = require_output(args.output)
    if not 4 <= args.samples <= 240:
        raise SystemExit("--samples must be in [4, 240]")
    if not 1 <= args.top_k <= 512:
        raise SystemExit("--top-k must be in [1, 512]")
    if not 0.0 <= args.minimum_influence <= 0.25:
        raise SystemExit("--minimum-influence must be in [0, 0.25]")

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
    rest_diagonal = float(np.linalg.norm(np.ptp(rest_vertices, axis=0)))
    rest_lengths = edge_lengths(rest_vertices, edges)
    valid_edges = rest_lengths > rest_diagonal * 1.0e-5
    semantics, bone_labels = semantic_labels(
        armature, rest_vertices, args.front_axis
    )
    influences = vertex_influences(body, bone_labels, args.minimum_influence)
    dominant, cross_limb_edges, mixed_vertices = edge_static_classification(
        edges, influences
    )

    available = list(bpy.data.actions)
    action = next(
        (
            candidate
            for candidate in available
            if args.action.lower() in candidate.name.lower()
        ),
        None,
    )
    if action is None:
        raise SystemExit(
            f"missing action {args.action}; available={[item.name for item in available]}"
        )
    armature.data.pose_position = "POSE"
    armature.animation_data_create()
    armature.animation_data.action = action
    start, end = action.frame_range
    candidates = []
    frame_summaries = []
    for sampled in np.linspace(start, end, args.samples):
        frame = int(round(float(sampled)))
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        posed_vertices, current_edges, current_faces = evaluated_geometry(body, depsgraph)
        if not np.array_equal(current_edges, edges) or not np.array_equal(
            current_faces, faces
        ):
            raise SystemExit("evaluated topology changed across animation frames")
        posed_lengths = edge_lengths(posed_vertices, edges)
        extension = np.full(len(edges), -np.inf, dtype=np.float64)
        extension[valid_edges] = np.maximum(
            posed_lengths[valid_edges] - rest_lengths[valid_edges], 0.0
        ) / rest_diagonal
        count = min(args.top_k, int(np.count_nonzero(valid_edges)))
        indices = np.argpartition(extension, -count)[-count:]
        indices = indices[np.argsort(extension[indices])[::-1]]
        frame_summaries.append(
            {
                "evaluated_frame": frame,
                "maximum_edge_extension_ratio_of_rest_diagonal": float(
                    extension[indices[0]]
                ),
                "cross_limb_edges_over_0_04": int(
                    np.count_nonzero(extension[cross_limb_edges] > 0.04)
                ),
            }
        )
        for edge_index in indices:
            candidates.append(
                edge_record(
                    edge_index=int(edge_index),
                    frame=frame,
                    rest_vertices=rest_vertices,
                    posed_vertices=posed_vertices,
                    edges=edges,
                    rest_lengths=rest_lengths,
                    posed_lengths=posed_lengths,
                    rest_diagonal=rest_diagonal,
                    influences=influences,
                    dominant=dominant,
                )
            )

    candidates.sort(
        key=lambda item: item["positive_extension_ratio_of_rest_diagonal"],
        reverse=True,
    )
    worst = candidates[: args.top_k]
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "action": {
            "requested": args.action,
            "resolved": action.name,
            "frame_range": [float(start), float(end)],
            "samples": args.samples,
        },
        "front_axis": args.front_axis,
        "rest_geometry": {
            "vertices": len(rest_vertices),
            "edges": len(edges),
            "triangles": len(faces),
            "diagonal": rest_diagonal,
        },
        "semantic_rig": {
            "root": semantics.root,
            "chains": {
                name: list(value) for name, value in semantics.chains().items()
            },
        },
        "static_weight_topology": {
            "minimum_recorded_influence": args.minimum_influence,
            "cross_limb_dominant_edge_count": int(len(cross_limb_edges)),
            "vertices_mixing_multiple_limb_chains_count": len(mixed_vertices),
            "vertices_mixing_multiple_limb_chains_sample": mixed_vertices[:256],
        },
        "sampled_frames": frame_summaries,
        "worst_edges": worst,
        "worst_edge_summary": {
            "maximum_extension_ratio_of_rest_diagonal": worst[0][
                "positive_extension_ratio_of_rest_diagonal"
            ],
            "cross_limb_dominant_edges_in_top_k": sum(
                item["cross_limb_dominant_edge"] for item in worst
            ),
            "top_k": len(worst),
        },
        "interpretation": {
            "cross_limb_dominant_edge": (
                "one native topology edge joins vertices dominated by different "
                "locomotion limbs; large animated extension is not repairable by "
                "global yaw and usually requires topology separation or regeneration"
            ),
            "mixed_limb_vertex": (
                "one native vertex receives material weights from more than one "
                "locomotion limb; weight sanitation may be sufficient when the "
                "underlying topology is otherwise separated"
            ),
        },
        "formal_dataset_registration_authorized": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "SKINNED_DEFORMATION_DIAGNOSTIC_OK "
        f"max_extension={payload['worst_edge_summary']['maximum_extension_ratio_of_rest_diagonal']:.6f} "
        f"cross_limb_top={payload['worst_edge_summary']['cross_limb_dominant_edges_in_top_k']} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
