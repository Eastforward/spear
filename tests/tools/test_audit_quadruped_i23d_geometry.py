import math

import numpy as np
import pytest

from tools import audit_quadruped_i23d_geometry as audit


def test_position_indexed_topology_ignores_duplicate_gltf_seam_vertices():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    faces = np.array(
        [[4, 1, 2], [0, 3, 1], [0, 2, 3], [5, 3, 2]], dtype=int
    )

    result = audit.position_indexed_topology(vertices, faces)

    assert result["position_unique_vertices"] == 4
    assert result["boundary_edges"] == 0
    assert result["nonmanifold_edges_over_two_faces"] == 0


def _synthetic_torso(center_z):
    rng = np.random.default_rng(7)
    points = []
    for x in np.linspace(-1.0, 1.0, 80):
        for _ in range(80):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            y = 0.5 + 0.3 * math.sin(theta)
            z = center_z(x) + 0.2 * math.cos(theta)
            points.append((x, y, z))
    return np.asarray(points)


def test_torso_midline_records_rigid_yaw_without_calling_it_bend():
    target_degrees = 17.0
    slope = math.tan(math.radians(target_degrees))

    result = audit.torso_midline_yaw(
        _synthetic_torso(lambda x: slope * x)
    )

    assert result["yaw_degrees"] == pytest.approx(target_degrees, abs=0.5)
    assert result["global_axis_yaw_degrees"] == pytest.approx(
        target_degrees, abs=0.5
    )
    assert result["centerline_bend_p95_degrees"] < 1.0


def test_torso_midline_detects_curvature_after_removing_rigid_yaw():
    slope = math.tan(math.radians(17.0))

    result = audit.torso_midline_yaw(
        _synthetic_torso(lambda x: slope * x + 0.35 * x * x)
    )

    assert result["global_axis_yaw_degrees"] == pytest.approx(17.0, abs=1.0)
    assert result["centerline_bend_p95_degrees"] > 10.0


def test_decision_does_not_reject_rigid_yaw_but_rejects_nonmanifold_ratio():
    result = audit.decision(
        {
            "nonmanifold_edge_ratio_per_triangle": 0.002,
        },
        {
            "yaw_degrees": 17.0,
            "global_axis_yaw_degrees": 17.0,
            "centerline_bend_p95_degrees": 0.2,
        },
    )

    assert result["status"] == "reject_before_lod_and_binding"
    assert result["rejection_reasons"] == [
        "nonmanifold_edge_ratio_exceeds_0_001"
    ]
    assert result["global_yaw_is_a_rejection_criterion"] is False


def test_decision_rejects_real_centerline_bend():
    result = audit.decision(
        {"nonmanifold_edge_ratio_per_triangle": 0.0},
        {
            "yaw_degrees": 0.0,
            "global_axis_yaw_degrees": 0.0,
            "centerline_bend_p95_degrees": 17.0,
        },
    )

    assert result["status"] == "reject_before_lod_and_binding"
    assert result["rejection_reasons"] == [
        "torso_centerline_bend_exceeds_10_degrees"
    ]
