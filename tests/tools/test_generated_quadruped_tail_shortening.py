"""Pure contract tests for generated-mesh local tail shortening."""

from __future__ import annotations

import copy

import pytest

from tools import generated_quadruped_tail_shortening as tail


def profile():
    return {
        "schema": tail.SCHEMA,
        "state_classification": "technical_spike_only",
        "formal_dataset_registration_authorized": False,
        "source": {
            "path": "/evidence/generated.glb",
            "sha256": "a" * 64,
            "size_bytes": 123,
        },
        "coordinate_space": "blender_world",
        "tail_corridor": {
            "centerline_world": [
                [0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0],
                [-0.7, 0.0, -0.2],
                [-1.0, 0.0, -0.4],
            ],
            "selection_radius_world": 0.12,
            "base_keep_arc_length_world": 0.1,
        },
        "shortening": {
            "free_tail_length_ratio": 0.05,
            "radial_thickness_ratio": 0.7,
        },
        "gates": {
            "minimum_moved_vertices": 3,
            "maximum_moved_vertex_fraction": 0.2,
            "minimum_initial_free_length_world": 0.5,
            "maximum_final_free_length_world": 0.08,
        },
        "authority": {
            "generated_mesh_is_geometry_authority": True,
            "template_geometry_substitution_forbidden": True,
            "non_tail_vertex_positions_must_remain_exact": True,
            "mesh_topology_must_remain_exact": True,
            "uv_and_pbr_must_remain_exact_in_memory": True,
        },
    }


def test_profile_is_fail_closed_and_never_authorizes_registration():
    parsed = tail.load_profile(profile())

    assert parsed.initial_free_length_world > 0.5
    assert parsed.final_free_length_world < 0.08

    changed = profile()
    changed["formal_dataset_registration_authorized"] = True
    with pytest.raises(tail.TailShorteningError, match="cannot authorize"):
        tail.load_profile(changed)

    changed = profile()
    changed["authority"]["template_geometry_substitution_forbidden"] = False
    with pytest.raises(tail.TailShorteningError, match="authority guards"):
        tail.load_profile(changed)


def test_mapping_keeps_root_and_every_point_outside_the_reviewed_corridor_exact():
    parsed = tail.load_profile(profile())

    root = tail.map_point(parsed, [-0.05, 0.02, 0.0])
    body = tail.map_point(parsed, [-0.6, 0.5, 0.0])

    assert root.selected is True
    assert root.moved is False
    assert root.output_world == root.source_world
    assert body.selected is False
    assert body.moved is False
    assert body.output_world == body.source_world


def test_mapping_shortens_only_the_free_shaft_along_the_reviewed_centerline():
    parsed = tail.load_profile(profile())
    source_tip = tail.point_at_arc_length(
        parsed, parsed.total_arc_length_world
    )

    mapped = tail.map_point(parsed, source_tip)
    expected_arc = (
        parsed.base_keep_arc_length_world + parsed.final_free_length_world
    )

    assert mapped.selected is True
    assert mapped.moved is True
    assert mapped.output_world == pytest.approx(
        tail.point_at_arc_length(parsed, expected_arc)
    )


def test_mapping_retains_a_bounded_fraction_of_cross_section_thickness():
    parsed = tail.load_profile(profile())
    source_center = tail.point_at_arc_length(
        parsed, parsed.total_arc_length_world
    )
    offset_tip = [source_center[0], source_center[1] + 0.1, source_center[2]]

    mapped = tail.map_point(parsed, offset_tip)
    output_center = tail.point_at_arc_length(
        parsed,
        parsed.base_keep_arc_length_world + parsed.final_free_length_world,
    )

    assert mapped.output_world[1] - output_center[1] == pytest.approx(0.07)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["tail_corridor"].update(
                {"base_keep_arc_length_world": 10.0}
            ),
            "keep length",
        ),
        (
            lambda value: value["shortening"].update(
                {"free_tail_length_ratio": 0.9}
            ),
            "free_tail_length_ratio",
        ),
        (
            lambda value: value["gates"].update(
                {"maximum_moved_vertex_fraction": 0.9}
            ),
            "maximum_moved_vertex_fraction",
        ),
    ],
)
def test_profile_rejects_unbounded_edits(mutate, message):
    value = copy.deepcopy(profile())
    mutate(value)

    with pytest.raises(tail.TailShorteningError, match=message):
        tail.load_profile(value)
