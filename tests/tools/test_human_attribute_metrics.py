from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tools import human_attribute_masks as metrics


def _mask(size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    core = np.zeros((size[1], size[0]), dtype=np.uint8)
    core[8:-8, 8:-8] = 255
    band = np.zeros_like(core)
    return Image.fromarray(core, "L"), Image.fromarray(band, "L")


def test_height_metrics_reject_an_unchanged_tall_man_candidate():
    size = (64, 96)
    source = Image.new("RGB", size, (120, 100, 80))
    alpha = np.zeros((size[1], size[0]), dtype=np.uint8)
    alpha[10:90, 20:44] = 255
    source_alpha = Image.fromarray(alpha, "L")
    core, band = _mask(size)

    result = metrics.evaluate_candidate_metrics(
        "tall_man",
        source=source,
        candidate=source.copy(),
        source_alpha=source_alpha,
        candidate_alpha=source_alpha.copy(),
        edit_core=core,
        transition_band=band,
    )

    assert result["passed"] is False
    assert result["checks"]["height_ratio"] is False
    assert result["metrics"]["height_ratio"] == 1.0


def test_glasses_metrics_require_both_lenses_and_the_nose_bridge():
    size = (128, 192)
    source = Image.new("RGB", size, (150, 130, 115))
    candidate_values = np.asarray(source).copy()
    candidate_values[25:32, 55:63] = (20, 20, 20)
    candidate = Image.fromarray(candidate_values, "RGB")
    alpha = Image.new("L", size, 255)
    core, band = _mask(size)

    result = metrics.evaluate_candidate_metrics(
        "glasses",
        source=source,
        candidate=candidate,
        source_alpha=alpha,
        candidate_alpha=alpha.copy(),
        edit_core=core,
        transition_band=band,
    )

    assert result["passed"] is False
    assert result["checks"]["bilateral_eye_change"] is False
    assert result["checks"]["bridge_occupancy"] is False


def test_hat_metrics_require_one_new_component_touching_the_head():
    size = (128, 192)
    source = Image.new("RGB", size, (140, 120, 100))
    alpha_values = np.zeros((size[1], size[0]), dtype=np.uint8)
    alpha_values[25:180, 42:86] = 255
    alpha = Image.fromarray(alpha_values, "L")
    core, band = _mask(size)

    result = metrics.evaluate_candidate_metrics(
        "hat",
        source=source,
        candidate=source.copy(),
        source_alpha=alpha,
        candidate_alpha=alpha.copy(),
        edit_core=core,
        transition_band=band,
    )

    assert result["passed"] is False
    assert result["checks"]["added_component_count"] is False
    assert result["checks"]["must_contact_head"] is False


def test_color_metrics_reject_an_unchanged_short_sleeve_candidate():
    size = (96, 128)
    source = Image.new("RGB", size, (25, 75, 45))
    alpha = Image.new("L", size, 255)
    core, band = _mask(size)

    result = metrics.evaluate_candidate_metrics(
        "short_sleeve_color",
        source=source,
        candidate=source.copy(),
        source_alpha=alpha,
        candidate_alpha=alpha.copy(),
        edit_core=core,
        transition_band=band,
    )

    assert result["passed"] is False
    assert result["checks"]["semantic_core_changed_fraction"] is False
    assert result["checks"]["target_median_delta_e"] is False
    assert result["metrics"]["source_alpha_byte_identical"] is True


@pytest.mark.parametrize(
    ("case_id", "pair_check", "extra_metric"),
    [
        ("trousers", "each_leg_changed_fraction", "luminance_correlation"),
        ("shoes", "each_shoe_changed_fraction", "sole_contact_y_delta_px"),
    ],
)
def test_lower_item_metrics_require_both_sides(case_id, pair_check, extra_metric):
    size = (96, 128)
    source = Image.new("RGB", size, (70, 70, 75))
    alpha = Image.new("L", size, 255)
    core, band = _mask(size)

    result = metrics.evaluate_candidate_metrics(
        case_id,
        source=source,
        candidate=source.copy(),
        source_alpha=alpha,
        candidate_alpha=alpha.copy(),
        edit_core=core,
        transition_band=band,
    )

    assert result["passed"] is False
    assert result["checks"][pair_check] is False
    assert extra_metric in result["metrics"]
