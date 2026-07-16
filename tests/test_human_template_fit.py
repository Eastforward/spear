"""Pure NumPy tests for stable-template fitting and texture projection."""
import ast
from pathlib import Path

import numpy as np
import pytest

from tools.human_part_transfer import HumanRegion
from tools.human_template_fit import (
    clamp_xy_displacements,
    dilate_unpainted,
    rasterize_uv_triangle,
    recolor_regions_preserve_luminance,
    regularize_region_labels_by_island,
    region_color_medians,
    region_palette_from_uv_samples,
    sample_texture_bilinear,
    smooth_xy_displacements,
    triangle_barycentric_3d,
)


def test_fit_never_changes_height_or_fixed_opacity_vertices():
    raw = np.array(((0.08, 0.00, 0.50), (0.01, 0.02, -0.25)))

    fitted = clamp_xy_displacements(raw, max_distance=0.04)
    smoothed = smooth_xy_displacements(
        fitted,
        ((1,), (0,)),
        np.array((False, True)),
        iterations=3,
        blend=0.5,
    )

    assert np.allclose(smoothed[:, 2], 0.0)
    assert np.allclose(smoothed[1], 0.0)


def test_clamp_uses_radial_xy_distance():
    raw = np.array(((3.0, 4.0, 9.0), (0.5, 0.0, -2.0)))

    fitted = clamp_xy_displacements(raw, max_distance=2.0)

    assert np.allclose(fitted[0], (1.2, 1.6, 0.0))
    assert np.allclose(fitted[1], (0.5, 0.0, 0.0))


def test_smoothing_uses_connected_fixed_vertices_as_zero_boundaries():
    displacements = np.array(
        ((1.0, 0.0, 4.0), (0.0, 0.0, 5.0), (8.0, 8.0, 6.0))
    )

    smoothed = smooth_xy_displacements(
        displacements,
        ((1,), (0, 2), (1,)),
        np.array((False, False, True)),
        iterations=1,
        blend=1.0,
    )

    assert np.allclose(
        smoothed,
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


def test_triangle_barycentric_3d_sums_to_one_on_an_edge():
    triangle = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))

    weights = triangle_barycentric_3d(np.array((1.0, 0.0, 3.0)), triangle)

    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, (0.5, 0.5, 0.0))


def test_texture_sampling_uses_blender_bottom_left_uv_origin():
    image = np.array(((10.0, 20.0), (30.0, 40.0)))

    assert sample_texture_bilinear(image, (0.0, 1.0)) == pytest.approx(10.0)
    assert sample_texture_bilinear(image, (0.0, 0.0)) == pytest.approx(30.0)


def test_texture_sampling_is_bilinear_for_rgb_and_single_channel_pbr():
    rgb = np.array(
        (
            ((0.0, 10.0, 20.0), (10.0, 20.0, 30.0)),
            ((20.0, 30.0, 40.0), (30.0, 40.0, 50.0)),
        )
    )
    pbr = np.array(((0.0, 0.2), (0.6, 1.0)))

    assert np.allclose(
        sample_texture_bilinear(rgb, (0.5, 0.5)),
        (15.0, 25.0, 35.0),
    )
    assert sample_texture_bilinear(pbr, (0.5, 0.5)) == pytest.approx(0.45)


def test_uv_triangle_rasterization_samples_source_uv_and_marks_pixels():
    image = np.zeros((4, 4, 3), dtype=np.float64)
    mask = np.zeros((4, 4), dtype=bool)
    source_image = np.array(
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        )
    )

    rasterize_uv_triangle(
        image,
        mask,
        target_uv=np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        source_uv=np.array(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        source_image=source_image,
        target_region=HumanRegion.LEFT_FOREARM,
        source_region=HumanRegion.LEFT_FOREARM,
    )

    assert mask.any()
    assert mask[0, 0]
    assert not mask[-1, -1]
    assert np.allclose(image[mask], (1.0, 0.0, 0.0))


def test_uv_triangle_rasterization_rejects_incompatible_body_region():
    image = np.zeros((2, 2, 3), dtype=np.float64)
    mask = np.zeros((2, 2), dtype=bool)
    triangle_uv = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises(ValueError, match="incompatible"):
        rasterize_uv_triangle(
            image,
            mask,
            triangle_uv,
            triangle_uv,
            image,
            target_region=HumanRegion.LEFT_FOREARM,
            source_region=HumanRegion.RIGHT_FOREARM,
        )

    assert not mask.any()


def test_dilation_grows_uv_island_into_four_neighbor_unpainted_pixels():
    image = np.zeros((3, 3, 3), dtype=np.float64)
    image[1, 1] = (0.2, 0.4, 0.6)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    dilate_unpainted(image, mask, iterations=1)

    expected_mask = np.array(
        ((False, True, False), (True, True, True), (False, True, False))
    )
    assert np.array_equal(mask, expected_mask)
    assert np.allclose(image[mask], (0.2, 0.4, 0.6))


def test_region_palette_uses_robust_median_and_requires_every_region():
    colors = np.array(
        (
            (0.1, 0.8, 0.2),
            (0.2, 0.7, 0.3),
            (1.0, 0.0, 1.0),
            (0.6, 0.5, 0.4),
        )
    )
    regions = np.array((0, 0, 0, 1))

    palette = region_color_medians(colors, regions, expected_regions=(0, 1))

    assert np.allclose(palette[0], (0.2, 0.7, 0.3))
    assert np.allclose(palette[1], (0.6, 0.5, 0.4))
    with pytest.raises(ValueError, match="missing region 2"):
        region_color_medians(colors, regions, expected_regions=(0, 1, 2))


def test_region_palette_samples_locked_template_correspondence_uvs():
    image = np.array(
        (
            ((0.9, 0.1, 0.1), (0.1, 0.8, 0.2)),
            ((0.2, 0.3, 0.9), (0.7, 0.7, 0.7)),
        ),
        dtype=np.float64,
    )
    source_uvs = np.array(
        (
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        dtype=np.float64,
    )
    template_regions = np.array((0, 0, 1, 1), dtype=np.int64)

    palette = region_palette_from_uv_samples(
        image,
        source_uvs,
        template_regions,
        expected_regions=(0, 1),
    )

    assert np.allclose(palette[0], (0.5, 0.45, 0.15))
    assert np.allclose(palette[1], (0.45, 0.5, 0.8))


def test_region_labels_use_one_dominant_region_per_disconnected_uv_island():
    labels = np.array(
        (
            (2, 2, -1, 7, 7),
            (2, 5, -1, 7, 4),
            (-1, -1, -1, 7, 7),
        ),
        dtype=np.int64,
    )
    mask = labels >= 0

    regularized = regularize_region_labels_by_island(labels, mask)

    assert np.array_equal(
        regularized,
        np.array(
            (
                (2, 2, -1, 7, 7),
                (2, 2, -1, 7, 7),
                (-1, -1, -1, 7, 7),
            ),
            dtype=np.int64,
        ),
    )


def test_region_recolor_preserves_local_texture_contrast_and_unlabelled_pixels():
    original = np.array(
        (
            ((0.20, 0.10, 0.10), (0.40, 0.20, 0.20)),
            ((0.10, 0.30, 0.10), (0.70, 0.60, 0.50)),
        ),
        dtype=np.float64,
    )
    labels = np.array(((0, 0), (1, -1)), dtype=np.int64)
    palette = {
        0: np.array((0.10, 0.40, 0.10)),
        1: np.array((0.20, 0.20, 0.80)),
    }

    recolored = recolor_regions_preserve_luminance(
        original, labels, palette, strength=1.0
    )

    assert np.allclose(recolored[0, 1], 2.0*recolored[0, 0])
    assert np.allclose(recolored[1, 1], original[1, 1])
    assert recolored[1, 0, 2] > recolored[1, 0, 0]


def test_module_does_not_import_runtime_or_heavy_geometry_dependencies():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "human_template_fit.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported_roots.isdisjoint({"bpy", "scipy", "cv2", "hy3dgen"})
