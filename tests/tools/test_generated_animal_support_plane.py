"""Executable contract tests for dual-authority quadruped support planes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.generated_animal_support_plane import (  # noqa: E402
    CAPTURE_RADIUS_RATIO,
    CROSSCHECK_METHOD,
    EVIDENCE_SCHEMA,
    PRIMARY_METHOD,
    SupportPlaneContractError,
    evaluate_dual_authority_support_plane,
)
from tools.generated_animal_support_plane_contract import (  # noqa: E402
    RIGID_TRANSFORM_ABSOLUTE_TOLERANCE_RATIO,
    validate_dual_authority_evidence,
    validate_rigid_leveling_transform,
)


MESH_DIAGONAL = 4.0


def synthetic_quadruped(*, cat_tail_offset: bool):
    centers = np.asarray(
        [
            [1.0, -0.5],
            [1.0, 0.5],
            [-1.0, -0.5],
            [-1.0, 0.5],
        ],
        dtype=np.float64,
    )
    heads = np.column_stack((centers, np.full(4, 0.2)))
    tails = np.column_stack((centers, np.zeros(4)))
    if cat_tail_offset:
        tails[2, 0] = -1.8
        tails[3, 0] = -1.8
    else:
        heads[:, 0] -= 0.08
        tails[:, 0] += 0.08

    vertices = []
    scores = []
    grid = (-0.03, -0.01, 0.01, 0.03)
    for index, center in enumerate(centers):
        floor = 0.01 * center[0] + 0.02 * center[1]
        for dx in grid:
            for dy in grid:
                vertices.append([center[0] + dx, center[1] + dy, floor])
                score = [0.0] * 4
                score[index] = 1.0
                scores.append(score)
    return (
        np.asarray(vertices, dtype=np.float64),
        heads,
        tails,
        np.asarray(scores, dtype=np.float64),
    )


def evaluate_fixture(*, cat_tail_offset: bool):
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=cat_tail_offset
    )
    return evaluate_dual_authority_support_plane(
        vertices,
        heads,
        tails,
        scores,
        mesh_diagonal=MESH_DIAGONAL,
    )


def _rotation_between(source, target):
    source = np.asarray(source, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.asarray(target, dtype=np.float64)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1.0e-15:
        return np.eye(3, dtype=np.float64)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def serialized_leveling(
    evidence,
    *,
    reverse_rotation=False,
    crosscheck_reverse_rotation=False,
    land_highest_foot=False,
):
    normal = np.asarray(
        evidence["primary"]["plane"]["normal"], dtype=np.float64
    )
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rotation = _rotation_between(
        up if reverse_rotation else normal,
        normal if reverse_rotation else up,
    )
    crosscheck_rotation = (
        _rotation_between(up, normal)
        if crosscheck_reverse_rotation
        else rotation
    )
    primary = (
        np.asarray(evidence["primary"]["foot_points"], dtype=np.float64)
        @ rotation.T
    )
    crosscheck = (
        np.asarray(evidence["crosscheck"]["foot_points"], dtype=np.float64)
        @ crosscheck_rotation.T
    )
    if land_highest_foot:
        translation = -float(primary[:, 2].max())
    else:
        translation = -float(primary[:, 2].min())
    primary[:, 2] += translation
    crosscheck[:, 2] += translation
    return {
        "applied_vertical_translation": translation,
        "foot_points_after": primary.tolist(),
        "crosscheck_foot_points_after": crosscheck.tolist(),
        "minimum_foot_z_after": float(primary[:, 2].min()),
    }


def test_cat_style_leaf_tail_offset_uses_complete_segments_without_fallback():
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    radius = MESH_DIAGONAL * CAPTURE_RADIUS_RATIO
    assert np.linalg.norm(vertices[32, :2] - tails[2, :2]) > radius
    assert np.linalg.norm(vertices[48, :2] - tails[3, :2]) > radius

    evidence = evaluate_dual_authority_support_plane(
        vertices,
        heads,
        tails,
        scores,
        mesh_diagonal=MESH_DIAGONAL,
    )

    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["fallback_used"] is False
    assert evidence["primary"]["method"] == PRIMARY_METHOD
    assert evidence["crosscheck"]["method"] == CROSSCHECK_METHOD
    assert evidence["primary"]["capture_counts"] == [16, 16, 16, 16]
    assert evidence["primary"]["contact_band_sizes"] == [16, 16, 16, 16]
    assert evidence["crosscheck"]["capture_counts"] == [16, 16, 16, 16]
    assert evidence["agreement"]["maximum_floor_z_absolute_delta"] == 0.0
    assert evidence["primary"]["plane"]["tilt_deg"] < 2.0


def test_shiba_style_aligned_segments_preserve_the_same_four_support_feet():
    evidence = evaluate_fixture(cat_tail_offset=False)

    assert evidence["primary"]["foot_points"] == evidence["crosscheck"][
        "foot_points"
    ]
    assert evidence["primary"]["plane"]["tilt_deg"] == pytest.approx(
        evidence["crosscheck"]["plane"]["tilt_deg"], abs=1.0e-12
    )
    validate_dual_authority_evidence(evidence)


def test_sparse_final_bottom_band_rejects_even_when_corridor_capture_is_dense():
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    vertices[:7, 2] += 0.02

    with pytest.raises(
        SupportPlaneContractError,
        match="bottom band captured fewer than 10 vertices",
    ):
        evaluate_dual_authority_support_plane(
            vertices,
            heads,
            tails,
            scores,
            mesh_diagonal=MESH_DIAGONAL,
        )


def test_fixed_corridor_does_not_expand_to_rescue_a_missing_foot():
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    vertices[:16, 1] += 0.3

    with pytest.raises(
        SupportPlaneContractError,
        match="primary semantic foot 0 captured fewer than 10 vertices",
    ):
        evaluate_dual_authority_support_plane(
            vertices,
            heads,
            tails,
            scores,
            mesh_diagonal=MESH_DIAGONAL,
        )


def test_weight_crosscheck_is_mandatory_and_never_replaces_primary_geometry():
    vertices, heads, tails, _scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    elevated = vertices.copy()
    elevated[:, 2] += 0.02
    combined = np.vstack((vertices, elevated))
    scores = np.zeros((len(combined), 4), dtype=np.float64)
    for index in range(4):
        start = len(vertices) + index * 16
        scores[start : start + 16, index] = 1.0

    with pytest.raises(
        SupportPlaneContractError,
        match="foot floors disagree beyond contact band",
    ):
        evaluate_dual_authority_support_plane(
            combined,
            heads,
            tails,
            scores,
            mesh_diagonal=MESH_DIAGONAL,
        )


def test_same_height_but_remote_weight_contact_is_rejected_by_xy_agreement():
    vertices, heads, tails, _scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    remote = vertices.copy()
    remote[:, 1] += 0.25
    combined = np.vstack((vertices, remote))
    scores = np.zeros((len(combined), 4), dtype=np.float64)
    for index in range(4):
        start = len(vertices) + index * 16
        scores[start : start + 16, index] = 1.0

    with pytest.raises(
        SupportPlaneContractError,
        match="contact centroids disagree beyond the fixed corridor",
    ):
        evaluate_dual_authority_support_plane(
            combined,
            heads,
            tails,
            scores,
            mesh_diagonal=MESH_DIAGONAL,
        )


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("vertices", "non-finite"),
        ("heads", "non-finite"),
        ("tails", "non-finite"),
        ("scores", "non-finite"),
    ],
)
def test_nonfinite_inputs_fail_closed(target, match):
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    values = {
        "vertices": vertices,
        "heads": heads,
        "tails": tails,
        "scores": scores,
    }
    values[target] = values[target].copy()
    values[target].flat[0] = np.nan

    with pytest.raises(SupportPlaneContractError, match=match):
        evaluate_dual_authority_support_plane(
            values["vertices"],
            values["heads"],
            values["tails"],
            values["scores"],
            mesh_diagonal=MESH_DIAGONAL,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.__setitem__("fallback_used", True),
            "fallback is forbidden",
        ),
        (
            lambda value: value["primary"]["capture_counts"].__setitem__(0, 9),
            "primary capture is sparse",
        ),
        (
            lambda value: value["primary"]["plane"].__setitem__(
                "tilt_deg", 0.0
            ),
            "primary plane tilt_deg is internally inconsistent",
        ),
        (
            lambda value: value["agreement"].__setitem__(
                "maximum_floor_z_absolute_delta", 1.0
            ),
            "maximum floor agreement is internally inconsistent",
        ),
        (
            lambda value: value["thresholds"].__setitem__(
                "capture_radius_ratio_of_mesh_diagonal", 0.06
            ),
            "threshold capture_radius_ratio_of_mesh_diagonal",
        ),
    ],
)
def test_serialized_evidence_is_recomputed_instead_of_trusted(mutate, match):
    evidence = deepcopy(evaluate_fixture(cat_tail_offset=True))
    mutate(evidence)

    with pytest.raises(SupportPlaneContractError, match=match):
        validate_dual_authority_evidence(evidence)


@pytest.mark.parametrize("flat", [True, False])
def test_rigid_leveling_transform_recomputes_identity_and_tilted_planes(flat):
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    if flat:
        vertices[:, 2] = 0.0
    evidence = evaluate_dual_authority_support_plane(
        vertices,
        heads,
        tails,
        scores,
        mesh_diagonal=MESH_DIAGONAL,
    )
    support = serialized_leveling(evidence)

    result = validate_rigid_leveling_transform(evidence, support)

    assert result["minimum_primary_z_after"] == pytest.approx(0.0, abs=1.0e-15)
    if flat:
        assert result["rotation_from_primary_normal_to_positive_z"] == (
            np.eye(3).tolist()
        )
    else:
        assert evidence["primary"]["plane"]["tilt_deg"] > 0.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__(
            "applied_vertical_translation",
            value["applied_vertical_translation"] + 0.001,
        ),
        lambda value: value["foot_points_after"][0].__setitem__(
            0, value["foot_points_after"][0][0] + 0.001
        ),
        lambda value: value["crosscheck_foot_points_after"][0].__setitem__(
            1, value["crosscheck_foot_points_after"][0][1] + 0.001
        ),
        lambda value: value.__setitem__(
            "minimum_foot_z_after", value["minimum_foot_z_after"] + 0.001
        ),
    ],
)
def test_each_serialized_post_leveling_field_is_fail_closed(mutate):
    evidence = evaluate_fixture(cat_tail_offset=True)
    support = serialized_leveling(evidence)
    mutate(support)

    with pytest.raises(
        SupportPlaneContractError, match="rigid leveling transform"
    ):
        validate_rigid_leveling_transform(evidence, support)


def test_coherently_fabricated_post_leveling_fields_are_rejected():
    evidence = evaluate_fixture(cat_tail_offset=True)
    support = {
        "applied_vertical_translation": 999.0,
        "foot_points_after": [[999.0, 999.0, 999.0]] * 4,
        "crosscheck_foot_points_after": [[-999.0, -999.0, -999.0]] * 4,
        "minimum_foot_z_after": 999.0,
    }

    with pytest.raises(
        SupportPlaneContractError, match="rigid leveling transform"
    ):
        validate_rigid_leveling_transform(evidence, support)


@pytest.mark.parametrize(
    "support",
    [
        lambda evidence: serialized_leveling(
            evidence, reverse_rotation=True
        ),
        lambda evidence: serialized_leveling(
            evidence, crosscheck_reverse_rotation=True
        ),
    ],
)
def test_wrong_or_split_rigid_transforms_are_rejected(support):
    evidence = evaluate_fixture(cat_tail_offset=True)

    with pytest.raises(
        SupportPlaneContractError, match="rigid leveling transform"
    ):
        validate_rigid_leveling_transform(evidence, support(evidence))


def test_landing_the_highest_instead_of_lowest_foot_is_rejected():
    vertices, heads, tails, scores = synthetic_quadruped(
        cat_tail_offset=True
    )
    vertices[:16, 2] += 0.002
    evidence = evaluate_dual_authority_support_plane(
        vertices,
        heads,
        tails,
        scores,
        mesh_diagonal=MESH_DIAGONAL,
    )

    with pytest.raises(
        SupportPlaneContractError, match="rigid leveling transform"
    ):
        validate_rigid_leveling_transform(
            evidence,
            serialized_leveling(evidence, land_highest_foot=True),
        )


def test_rigid_transform_float_tolerance_has_a_fail_closed_boundary():
    evidence = evaluate_fixture(cat_tail_offset=True)
    support = serialized_leveling(evidence)
    scale = max(
        1.0,
        evidence["mesh_diagonal"],
        max(
            abs(value)
            for authority in ("primary", "crosscheck")
            for point in evidence[authority]["foot_points"]
            for value in point
        ),
    )
    tolerance = RIGID_TRANSFORM_ABSOLUTE_TOLERANCE_RATIO * scale

    within = deepcopy(support)
    within["applied_vertical_translation"] += tolerance * 0.5
    validate_rigid_leveling_transform(evidence, within)

    outside = deepcopy(support)
    outside["applied_vertical_translation"] += tolerance * 10.0
    with pytest.raises(
        SupportPlaneContractError, match="rigid leveling transform"
    ):
        validate_rigid_leveling_transform(evidence, outside)
