"""Functional contracts for bounded same-Pixal-mesh limb repair."""

from __future__ import annotations

import numpy as np

from tools.generated_quadruped_geometry_repair import (
    RepairSpec,
    audit_expected_surface_signatures,
    audit_immutable_source_topology,
    audit_low_slice_limb_chains,
    canonical_triangle_surface_signatures,
    identify_tail_surface,
    mirror_points_across_y,
    mirror_points_across_y_with_attachment_taper,
    repair_vertex_masks,
)


def _bounds():
    return np.asarray((0.0, -1.0, 0.0)), np.asarray((10.0, 1.0, 10.0))


def test_repair_masks_only_source_side_limb_corridors_and_protects_tail():
    spec = RepairSpec()
    lower, upper = _bounds()
    points = np.asarray(
        (
            (2.7, 0.5, 0.2),   # front donor foot
            (3.6, 0.5, 5.6),   # front donor attachment
            (8.3, 0.5, 0.2),   # hind donor foot
            (6.8, 0.5, 5.6),   # hind donor attachment
            (2.7, -0.5, 0.2),  # rejected far front
            (8.3, -0.5, 0.2),  # rejected far hind
            (8.5, 0.5, 1.0),   # protected low/downward tail surface
            (5.0, 0.5, 5.0),   # body outside corridors
            (3.6, 0.0, 5.6),   # central authenticated attachment bridge
        ),
        dtype=np.float64,
    )
    tail_surface = np.zeros(len(points), dtype=bool)
    tail_surface[6] = True
    masks = repair_vertex_masks(
        points,
        lower,
        upper,
        0.0,
        spec,
        tail_surface=tail_surface,
    )

    assert masks["donor"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks["tail_protected"][6]
    assert masks["preserved_central_attachment_bridge"][8]
    assert not masks["replace_corridor"][8]
    assert not np.any(masks["donor"] & masks["tail_protected"])
    assert not np.any(masks["replace_corridor"] & masks["tail_protected"])


def test_tail_surface_identification_is_source_graph_based_and_height_independent():
    spec = RepairSpec(tail_protection_x_fraction=0.78)
    lower, upper = _bounds()
    points = np.asarray(
        (
            (8.0, 0.0, 1.0),
            (9.0, 0.0, 1.1),
            (10.0, 0.0, 1.2),  # posterior low tail seed
            (8.2, -0.8, 0.2),  # disconnected hind foot in same x slab
            (8.3, -0.8, 0.5),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 10.0),
            (10.0, 1.0, 10.0),
        ),
        dtype=np.float64,
    )
    edges = ((0, 1), (1, 2), (3, 4), (5, 6), (6, 7))

    tail, record = identify_tail_surface(
        points,
        edges,
        lower,
        upper,
        spec,
        minimum_vertices=3,
    )

    assert tail.tolist() == [True, True, True, False, False, False, False, False]
    assert record["passed"]
    assert record["height_used_for_selection"] is False
    assert record["vertices_below_legacy_height_cut"] == 3
    assert record["seed_vertex_index"] == 2


def test_mirror_changes_only_lateral_coordinate():
    points = np.asarray(((1.0, 0.75, 2.0), (-1.0, 0.25, 3.0)))
    mirrored = mirror_points_across_y(points, plane_y=0.1)

    np.testing.assert_allclose(mirrored[:, 0], points[:, 0])
    np.testing.assert_allclose(mirrored[:, 2], points[:, 2])
    np.testing.assert_allclose(mirrored[:, 1], (-0.55, -0.05))


def test_attachment_taper_keeps_feet_exactly_mirrored_and_draws_only_top_inward():
    spec = RepairSpec()
    lower, upper = _bounds()
    points = np.asarray(
        (
            (2.7, 0.5, 1.0),
            (3.6, 0.5, spec.attachment_height_fraction * 10.0),
        )
    )
    mirrored, record = mirror_points_across_y_with_attachment_taper(
        points,
        plane_y=0.0,
        lower=lower,
        upper=upper,
        spec=spec,
    )

    assert mirrored[0, 1] == -0.5
    assert mirrored[1, 1] == -0.5 * spec.mirrored_attachment_top_lateral_scale
    np.testing.assert_allclose(mirrored[:, (0, 2)], points[:, (0, 2)])
    assert record["tapered_vertex_count"] == 1


def _synthetic_four_leg_mesh(with_low_membrane: bool):
    points = []
    edges = []
    for x in (2.7, 8.3):
        for y in (-0.5, 0.5):
            first = len(points)
            for index, z in enumerate(np.linspace(0.0, 0.7, 16)):
                points.append((x, y, float(z)))
                if index:
                    edges.append((first + index - 1, first + index))
    # Robust ceiling and longitudinal extents come from body points above the
    # 0.4 low-slice threshold. They do not join the induced lower graph.
    points.extend(
        (
            (0.0, -1.0, 1.0),
            (0.0, 1.0, 1.0),
            (10.0, -1.0, 1.0),
            (10.0, 1.0, 1.0),
        )
    )
    if with_low_membrane:
        edges.append((0, 16))
    return np.asarray(points, dtype=np.float64), edges


def test_low_slice_gate_accepts_four_independent_floor_to_attachment_chains():
    points, edges = _synthetic_four_leg_mesh(with_low_membrane=False)
    audit = audit_low_slice_limb_chains(
        points,
        edges,
        sagittal_plane_y=0.0,
        spec=RepairSpec(),
        minimum_component_vertices=4,
    )

    assert audit["passed"]
    assert audit["checks"]["no_low_cross_limb_membrane"]
    assert audit["quadrant_counts"] == {
        "front_negative_y": 1,
        "front_positive_y": 1,
        "hind_negative_y": 1,
        "hind_positive_y": 1,
    }


def test_low_slice_gate_rejects_cross_limb_membrane():
    points, edges = _synthetic_four_leg_mesh(with_low_membrane=True)
    audit = audit_low_slice_limb_chains(
        points,
        edges,
        sagittal_plane_y=0.0,
        spec=RepairSpec(),
        minimum_component_vertices=4,
    )

    assert not audit["passed"]
    assert not audit["checks"]["no_low_cross_limb_membrane"]
    assert (
        "low_slice_has_fewer_than_four_substantial_components"
        in audit["rejection_reasons"]
    )


def test_surface_identity_signature_pins_position_uv_material_and_winding():
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    uvs = np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),))
    materials = np.asarray((0,), dtype=np.int64)
    expected = canonical_triangle_surface_signatures(
        points,
        triangles,
        uvs,
        materials,
    )

    cyclic = canonical_triangle_surface_signatures(
        points,
        np.asarray(((1, 2, 0),)),
        np.asarray((((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),)),
        materials,
    )
    reversed_winding = canonical_triangle_surface_signatures(
        points,
        np.asarray(((0, 2, 1),)),
        np.asarray((((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),)),
        materials,
    )

    assert audit_expected_surface_signatures(expected, cyclic)["passed"]
    assert not audit_expected_surface_signatures(
        expected,
        reversed_winding,
    )["passed"]


def test_topology_preflight_rejects_boundary_outside_mutable_corridor():
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)

    rejected = audit_immutable_source_topology(
        points,
        triangles,
        np.asarray((False, False, False)),
    )
    locally_mutable = audit_immutable_source_topology(
        points,
        triangles,
        np.asarray((True, True, True)),
    )

    assert not rejected["passed"]
    assert rejected["outside_corridor_boundary_edges"] == 3
    # The defects are local-authority candidates, but the surface is still not
    # itself a valid closed output until a local weld/fill removes them.
    assert locally_mutable["outside_corridor_boundary_edges"] == 0
    assert locally_mutable[
        "inside_corridor_defective_edges_available_for_local_repair"
    ] == 3


def test_topology_preflight_accepts_closed_single_component_outside_corridor():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ),
        dtype=np.int64,
    )

    audit = audit_immutable_source_topology(
        points,
        triangles,
        np.zeros(len(points), dtype=bool),
    )

    assert audit["passed"]
    assert audit["logical_component_count"] == 1
    assert audit["outside_corridor_boundary_edges"] == 0
    assert audit["outside_corridor_edges_over_two_faces"] == 0
