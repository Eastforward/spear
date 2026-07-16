#!/usr/bin/env python3
"""Pure, source-hash-specific mask and alpha contracts for Route-2 edits."""

from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


MASK_CONSTRUCTION_VERSION = "human_attribute_source_semantics_v2"
MASK_AGENT_VISUAL_CHECKS = (
    "target_semantic_is_precise",
    "non_target_regions_are_protected",
    "transition_band_is_local_and_reasonable",
    "background_is_not_selected_except_required_additive_silhouette",
    "overlay_matches_the_approved_source",
)
MASK_AGENT_PASS = "agent_qa_passed_pending_user_acceptance"
TARGET_COLOR_SRGB = {
    "cobalt_blue": (0, 71, 171),
    "warm_beige": (196, 164, 132),
    "black": (18, 18, 18),
}


def _gates(**values: Any) -> dict[str, Any]:
    common = {
        "outside_band_changed_pixels_max": 0,
        "outside_band_max_abs_channel_delta": 0,
        "minimum_canvas_margin_px": 16,
        "agent_full_resolution_visual_qa_required": True,
    }
    common.update(values)
    return common


CASE_MASK_CONTRACTS: dict[str, dict[str, Any]] = {
    "tall_man": {
        "base_asset_id": "rocketbox_male_adult_01",
        "strategy": "foot_anchored_height_silhouette_union",
        "target_parameters": {"height_ratio": 1.09, "ratio_tolerance": 0.02},
        "transition_radius_px": 8,
        "quantitative_gates": _gates(
            foot_anchor_tolerance_px=4,
            center_x_tolerance_px=4,
            limb_gap_area_retention_min=0.90,
            inverse_aligned_face_mae_8bit_max=3.0,
            inverse_aligned_face_p99_8bit_max=12.0,
        ),
    },
    "short_woman": {
        "base_asset_id": "rocketbox_female_adult_01",
        "strategy": "foot_anchored_height_silhouette_union",
        "target_parameters": {"height_ratio": 0.90, "ratio_tolerance": 0.02},
        "transition_radius_px": 8,
        "quantitative_gates": _gates(
            foot_anchor_tolerance_px=4,
            center_x_tolerance_px=4,
            limb_gap_area_retention_min=0.90,
            inverse_aligned_face_mae_8bit_max=3.0,
            inverse_aligned_face_p99_8bit_max=12.0,
        ),
    },
    "glasses": {
        "base_asset_id": "rocketbox_male_adult_01",
        "strategy": "reviewed_eyewear_stroke_envelope",
        "reviewed_envelope_id": "male_seed41_eyewear_v1",
        "transition_radius_px": 8,
        "target_parameters": {"style": "thin_matte_black_rectangular"},
        "quantitative_gates": _gates(
            bilateral_eye_change_required=True,
            lens_center_y_delta_px_max=4,
            bridge_occupancy_required=True,
            protected_face_changed_pixels_max=0,
        ),
    },
    "hat": {
        "base_asset_id": "rocketbox_female_adult_01",
        "strategy": "reviewed_headwear_add_envelope",
        "reviewed_envelope_id": "female_seed73_cap_v1",
        "protected_below_y": 0.134,
        "transition_radius_px": 8,
        "target_parameters": {"style": "plain_navy_baseball_cap", "brim": "forward"},
        "quantitative_gates": _gates(
            added_component_count=1,
            must_contact_head=True,
            changed_pixels_below_brow_max=0,
        ),
    },
    "short_sleeve_color": {
        "base_asset_id": "rocketbox_male_adult_01",
        "strategy": "source_foreground_lab_connected_component",
        "roi": [0.25, 0.20, 0.75, 0.53],
        "seed_points": [[0.50, 0.32], [0.38, 0.31], [0.62, 0.31]],
        "lab_tolerance": 32.0,
        "expected_components": 1,
        "transition_radius_px": 4,
        "target_parameters": {"semantic": "upper_garment", "color": "cobalt_blue"},
        "quantitative_gates": _gates(
            source_alpha_byte_identical=True,
            semantic_core_changed_fraction_min=0.80,
            target_median_delta_e_max=10.0,
            luminance_correlation_min=0.98,
            edge_correlation_min=0.95,
        ),
    },
    "trousers": {
        "base_asset_id": "rocketbox_female_adult_01",
        "strategy": "source_foreground_lab_connected_component",
        "roi": [0.35, 0.50, 0.65, 0.89],
        "protected_above_y": 0.50,
        "seed_points": [[0.44, 0.60], [0.56, 0.60], [0.44, 0.80], [0.56, 0.80]],
        "lab_tolerance": 35.0,
        "expected_components": 1,
        "transition_radius_px": 4,
        "target_parameters": {"semantic": "full_length_trousers", "color": "warm_beige"},
        "quantitative_gates": _gates(
            source_alpha_byte_identical=True,
            each_leg_changed_fraction_min=0.80,
            target_median_delta_e_max=10.0,
            luminance_correlation_min=0.98,
            edge_correlation_min=0.95,
        ),
    },
    "shoes": {
        "base_asset_id": "rocketbox_male_adult_01",
        "strategy": "source_foreground_lab_connected_component",
        "roi": [0.32, 0.875, 0.68, 0.97],
        "seed_points": [[0.41, 0.92], [0.59, 0.92]],
        "lab_tolerance": 34.0,
        "expected_components": 2,
        "transition_radius_px": 3,
        "target_parameters": {"semantic": "shoes", "color": "black"},
        "quantitative_gates": _gates(
            source_alpha_byte_identical=True,
            each_shoe_changed_fraction_min=0.70,
            sole_contact_y_tolerance_px=1,
            edge_correlation_min=0.95,
        ),
    },
}


def _image_bytes_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _normalized_pixel(point: Sequence[float], size: tuple[int, int]) -> tuple[int, int]:
    if len(point) != 2 or any(not 0.0 <= float(value) <= 1.0 for value in point):
        raise ValueError("review annotation coordinates must be normalized 2D points")
    return (
        int(round(float(point[0]) * (size[0] - 1))),
        int(round(float(point[1]) * (size[1] - 1))),
    )


def _normalized_roi(rectangle: Sequence[float], size: tuple[int, int]) -> np.ndarray:
    if len(rectangle) != 4:
        raise ValueError("semantic ROI must contain four normalized coordinates")
    left, top = _normalized_pixel(rectangle[:2], size)
    right, bottom = _normalized_pixel(rectangle[2:], size)
    if not left < right or not top < bottom:
        raise ValueError("semantic ROI is empty or reversed")
    result = np.zeros((size[1], size[0]), dtype=bool)
    result[top : bottom + 1, left : right + 1] = True
    return result


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116 * transformed[..., 1] - 16,
            500 * (transformed[..., 0] - transformed[..., 1]),
            200 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    )


def _seeded_lab_semantic(
    source: Image.Image, alpha: Image.Image, contract: Mapping[str, Any]
) -> np.ndarray:
    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    foreground = np.asarray(alpha.convert("L"), dtype=np.uint8) >= 128
    roi = _normalized_roi(contract["roi"], source.size)
    lab = _srgb_to_lab(rgb)
    seed_pixels = [_normalized_pixel(point, source.size) for point in contract["seed_points"]]
    seed_lab = np.stack([lab[y, x] for x, y in seed_pixels], axis=0)
    distances = np.min(np.linalg.norm(lab[..., None, :] - seed_lab[None, None, :, :], axis=-1), axis=2)
    candidate = foreground & roi & (distances <= float(contract["lab_tolerance"]))
    candidate = ndimage.binary_closing(candidate, iterations=2)
    labels, _ = ndimage.label(candidate)
    selected: set[int] = set()
    for x, y in seed_pixels:
        label = int(labels[y, x])
        if label:
            selected.add(label)
    if not selected:
        raise ValueError("reviewed Lab seeds do not identify a foreground component")
    semantic = np.isin(labels, list(selected)) & foreground
    semantic = ndimage.binary_fill_holes(semantic) & foreground
    semantic = ndimage.binary_closing(semantic, iterations=1) & foreground
    final_labels, _ = ndimage.label(semantic)
    final_selected = {
        int(final_labels[y, x])
        for x, y in seed_pixels
        if int(final_labels[y, x]) != 0
    }
    semantic = np.isin(final_labels, list(final_selected)) & foreground
    return semantic


def _foot_anchored_scaled_mask(source_fg: np.ndarray, ratio: float) -> tuple[np.ndarray, int]:
    rows, columns = np.where(source_fg)
    if rows.size == 0:
        raise ValueError("approved source alpha has no foreground")
    foot_y = int(rows.max())
    target = np.zeros_like(source_fg)
    y, x = np.where(source_fg)
    target_y = foot_y - np.rint((foot_y - y) * ratio).astype(np.int64)
    valid = (target_y >= 0) & (target_y < source_fg.shape[0])
    target[target_y[valid], x[valid]] = True
    target = ndimage.binary_closing(target, iterations=1)
    return target, foot_y


def _reviewed_accessory_mask(case_id: str, size: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    scale = max(1, round(size[0] / 1152))
    metadata: dict[str, int] = {}
    if case_id == "glasses":
        left = [(0.449, 0.137), (0.493, 0.134), (0.495, 0.157), (0.451, 0.158), (0.449, 0.137)]
        right = [(0.505, 0.134), (0.550, 0.137), (0.548, 0.158), (0.504, 0.157), (0.505, 0.134)]
        for polygon in (left, right):
            draw.line([_normalized_pixel(point, size) for point in polygon], fill=255, width=8 * scale, joint="curve")
        draw.line(
            [_normalized_pixel((0.493, 0.144), size), _normalized_pixel((0.505, 0.144), size)],
            fill=255,
            width=6 * scale,
        )
        draw.line(
            [_normalized_pixel((0.449, 0.143), size), _normalized_pixel((0.432, 0.148), size)],
            fill=255,
            width=5 * scale,
        )
        draw.line(
            [_normalized_pixel((0.550, 0.143), size), _normalized_pixel((0.568, 0.148), size)],
            fill=255,
            width=5 * scale,
        )
    elif case_id == "hat":
        polygon = [
            (0.380, 0.125),
            (0.405, 0.112),
            (0.420, 0.074),
            (0.460, 0.049),
            (0.540, 0.049),
            (0.580, 0.074),
            (0.595, 0.112),
            (0.620, 0.125),
            (0.600, 0.132),
            (0.400, 0.132),
        ]
        draw.polygon([_normalized_pixel(point, size) for point in polygon], fill=255)
        metadata["protected_face_below_y_px"] = _normalized_pixel((0.5, 0.134), size)[1]
    else:
        raise ValueError(f"not an accessory mask case: {case_id}")
    return np.asarray(image, dtype=np.uint8) == 255, metadata


def transition_and_guard(core: Image.Image, *, radius_px: int) -> tuple[Image.Image, Image.Image]:
    values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    if not np.any(values) or np.all(values):
        raise ValueError("edit core must be nonempty and nonfull")
    dilated = ndimage.binary_dilation(values, iterations=int(radius_px))
    band = dilated & ~values
    guard = ~dilated
    return (
        Image.fromarray(np.where(band, 255, 0).astype(np.uint8), "L"),
        Image.fromarray(np.where(guard, 255, 0).astype(np.uint8), "L"),
    )


def construct_mask_bundle(
    case_id: str, source: Image.Image, source_alpha: Image.Image
) -> dict[str, Any]:
    if case_id not in CASE_MASK_CONTRACTS:
        raise ValueError(f"unknown Route-2 attribute case: {case_id}")
    source = source.convert("RGB")
    source_alpha = source_alpha.convert("L")
    if source.size != source_alpha.size:
        raise ValueError("source RGB and approved ISNet alpha canvas differ")
    contract = CASE_MASK_CONTRACTS[case_id]
    source_fg = np.asarray(source_alpha, dtype=np.uint8) >= 128
    metadata: dict[str, Any] = {}
    if contract["strategy"] == "foot_anchored_height_silhouette_union":
        target, foot_y = _foot_anchored_scaled_mask(
            source_fg, float(contract["target_parameters"]["height_ratio"])
        )
        core_values = source_fg | target
        metadata["foot_anchor_y_px"] = foot_y
    elif contract["strategy"] == "source_foreground_lab_connected_component":
        core_values = _seeded_lab_semantic(source, source_alpha, contract)
    else:
        core_values, accessory = _reviewed_accessory_mask(case_id, source.size)
        metadata.update(accessory)
        metadata["reviewed_envelope_id"] = contract["reviewed_envelope_id"]
    core = Image.fromarray(np.where(core_values, 255, 0).astype(np.uint8), "L")
    band, guard = transition_and_guard(core, radius_px=int(contract["transition_radius_px"]))
    if "protected_above_y" in contract:
        protected_above_y_px = int(
            round(float(contract["protected_above_y"]) * (source.height - 1))
        )
        band_values = np.asarray(band, dtype=np.uint8) == 255
        band_values[:protected_above_y_px] = False
        guard_values = ~(core_values | band_values)
        band = Image.fromarray(
            np.where(band_values, 255, 0).astype(np.uint8), "L"
        )
        guard = Image.fromarray(
            np.where(guard_values, 255, 0).astype(np.uint8), "L"
        )
        metadata["protected_above_y_px"] = protected_above_y_px
    if "protected_below_y" in contract:
        protected_below_y_px = int(
            round(float(contract["protected_below_y"]) * (source.height - 1))
        )
        band_values = np.asarray(band, dtype=np.uint8) == 255
        band_values[protected_below_y_px:] = False
        guard_values = ~(core_values | band_values)
        band = Image.fromarray(
            np.where(band_values, 255, 0).astype(np.uint8), "L"
        )
        guard = Image.fromarray(
            np.where(guard_values, 255, 0).astype(np.uint8), "L"
        )
        metadata["protected_below_y_px"] = protected_below_y_px
    _, count = ndimage.label(core_values)
    background = np.asarray(source_alpha, dtype=np.uint8) < 128
    metrics = {
        "source_alpha_sha256": _image_bytes_sha256(source_alpha),
        "core_pixels": int(np.count_nonzero(core_values)),
        "core_fraction": float(np.count_nonzero(core_values) / core_values.size),
        "core_background_fraction": float(
            np.count_nonzero(core_values & background) / np.count_nonzero(core_values)
        ),
        "connected_component_count": int(count),
        **metadata,
    }
    return {
        "construction_version": MASK_CONSTRUCTION_VERSION,
        "case_id": case_id,
        "base_asset_id": contract["base_asset_id"],
        "strategy": contract["strategy"],
        "target_parameters": dict(contract["target_parameters"]),
        "quantitative_gates": dict(contract["quantitative_gates"]),
        "edit_core": core,
        "transition_band": band,
        "protected_guard": guard,
        "metrics": metrics,
    }


def feathered_composite(
    source: Image.Image,
    generated: Image.Image,
    edit_core: Image.Image,
    transition_band: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    source_values = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_values = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    core = np.asarray(edit_core.convert("L"), dtype=np.uint8) == 255
    band = np.asarray(transition_band.convert("L"), dtype=np.uint8) == 255
    if source_values.shape != generated_values.shape or source_values.shape[:2] != core.shape or core.shape != band.shape:
        raise ValueError("source/generated/mask canvas mismatch")
    if np.any(core & band):
        raise ValueError("edit core and transition band overlap")
    guard = ~(core | band)
    weight = np.zeros(core.shape, dtype=np.float64)
    weight[core] = 1.0
    distance_to_core = ndimage.distance_transform_edt(~core)
    distance_to_guard = ndimage.distance_transform_edt(~guard)
    denominator = distance_to_core + distance_to_guard
    weight[band] = distance_to_guard[band] / np.maximum(denominator[band], 1e-9)
    result = np.rint(
        source_values.astype(np.float64) * (1.0 - weight[..., None])
        + generated_values.astype(np.float64) * weight[..., None]
    ).astype(np.uint8)
    changed = np.any(result != source_values, axis=2)
    outside_delta = np.abs(result.astype(np.int16) - source_values.astype(np.int16))[guard]
    return Image.fromarray(result, "RGB"), {
        "outside_changed_pixels": int(np.count_nonzero(changed & guard)),
        "outside_max_abs_channel_delta": int(outside_delta.max(initial=0)),
        "inside_changed_pixels": int(np.count_nonzero(changed & core)),
        "transition_changed_pixels": int(np.count_nonzero(changed & band)),
        "transition_is_feathered": bool(
            np.any(weight[band] > 0.0) and np.any(weight[band] < 1.0)
        ),
    }


def build_candidate_alpha(
    case_id: str,
    source_alpha: Image.Image,
    predicted_alpha: Image.Image | None,
    edit_core: Image.Image,
    transition_band: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    source = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    core = np.asarray(edit_core.convert("L"), dtype=np.uint8) == 255
    band = np.asarray(transition_band.convert("L"), dtype=np.uint8) == 255
    if source.shape != core.shape or core.shape != band.shape:
        raise ValueError("source alpha and mask canvas mismatch")
    if case_id in {"short_sleeve_color", "trousers", "shoes"}:
        return Image.fromarray(source.copy(), "L"), {
            "policy": "reuse_source_alpha_exact",
            "outside_changed_pixels": 0,
        }
    if predicted_alpha is None:
        raise ValueError(f"{case_id} requires a pinned ISNet candidate-alpha prediction")
    predicted = np.asarray(predicted_alpha.convert("L"), dtype=np.uint8)
    if predicted.shape != source.shape:
        raise ValueError("predicted alpha canvas mismatch")
    allowed = core | band
    result = source.copy()
    if case_id in {"glasses", "hat"}:
        result[allowed] = np.maximum(source[allowed], predicted[allowed])
        policy = "add_envelope_only"
    elif case_id in {"tall_man", "short_woman"}:
        result[allowed] = predicted[allowed]
        policy = "height_silhouette_rebuild"
    else:
        raise ValueError(f"unknown alpha policy for {case_id}")
    return Image.fromarray(result, "L"), {
        "policy": policy,
        "outside_changed_pixels": int(np.count_nonzero(result[~allowed] != source[~allowed])),
    }


def _foreground_bbox(values: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(values)
    if rows.size == 0:
        raise ValueError("candidate alpha has no foreground")
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def _inverse_align_height_candidate(
    candidate: Image.Image,
    candidate_alpha: Image.Image,
    source_bbox: tuple[int, int, int, int],
    candidate_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    sx0, sy0, sx1, sy1 = source_bbox
    cx0, cy0, cx1, cy1 = candidate_bbox
    target_size = (sx1 - sx0 + 1, sy1 - sy0 + 1)
    rgb_crop = candidate.convert("RGB").crop((cx0, cy0, cx1 + 1, cy1 + 1))
    alpha_crop = candidate_alpha.convert("L").crop((cx0, cy0, cx1 + 1, cy1 + 1))
    rgb = np.asarray(rgb_crop.resize(target_size, Image.Resampling.BICUBIC))
    alpha = np.asarray(alpha_crop.resize(target_size, Image.Resampling.NEAREST))
    return rgb, alpha


def _evaluate_height_metrics(
    case_id: str,
    *,
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
) -> dict[str, Any]:
    contract = CASE_MASK_CONTRACTS[case_id]
    gates = contract["quantitative_gates"]
    target = contract["target_parameters"]
    source_fg = np.asarray(source_alpha.convert("L"), dtype=np.uint8) >= 128
    candidate_fg = np.asarray(candidate_alpha.convert("L"), dtype=np.uint8) >= 128
    source_bbox = _foreground_bbox(source_fg)
    candidate_bbox = _foreground_bbox(candidate_fg)
    sx0, sy0, sx1, sy1 = source_bbox
    cx0, cy0, cx1, cy1 = candidate_bbox
    source_height = sy1 - sy0 + 1
    candidate_height = cy1 - cy0 + 1
    ratio = float(candidate_height / source_height)
    foot_delta = int(abs(cy1 - sy1))
    center_delta = float(abs((cx0 + cx1) / 2.0 - (sx0 + sx1) / 2.0))

    aligned_rgb, aligned_alpha = _inverse_align_height_candidate(
        candidate, candidate_alpha, source_bbox, candidate_bbox
    )
    source_crop = np.asarray(source.convert("RGB"))[sy0 : sy1 + 1, sx0 : sx1 + 1]
    source_crop_fg = source_fg[sy0 : sy1 + 1, sx0 : sx1 + 1]
    face_bottom = max(1, int(round(source_height * 0.22)))
    face_half_width = max(1, int(round((sx1 - sx0 + 1) * 0.16)))
    face_center = (sx1 - sx0 + 1) // 2
    face_slice = (
        slice(0, face_bottom),
        slice(max(0, face_center - face_half_width), min(source_crop.shape[1], face_center + face_half_width + 1)),
    )
    face_delta = np.abs(
        aligned_rgb[face_slice].astype(np.int16)
        - source_crop[face_slice].astype(np.int16)
    )
    face_mae = float(face_delta.mean()) if face_delta.size else 0.0
    face_p99 = float(np.percentile(face_delta, 99)) if face_delta.size else 0.0

    gap_top = int(round(source_height * 0.18))
    gap_bottom = max(gap_top + 1, int(round(source_height * 0.62)))
    gap_region = ~source_crop_fg[gap_top:gap_bottom]
    gap_count = int(np.count_nonzero(gap_region))
    retained = (
        float(
            np.count_nonzero(
                gap_region & ~(aligned_alpha[gap_top:gap_bottom] >= 128)
            )
            / gap_count
        )
        if gap_count
        else 1.0
    )
    tolerance = float(target["ratio_tolerance"])
    checks = {
        "height_ratio": abs(ratio - float(target["height_ratio"])) <= tolerance,
        "foot_anchor": foot_delta <= int(gates["foot_anchor_tolerance_px"]),
        "center_x": center_delta <= float(gates["center_x_tolerance_px"]),
        "limb_gap_area_retention": retained
        >= float(gates["limb_gap_area_retention_min"]),
        "inverse_aligned_face_mae": face_mae
        <= float(gates["inverse_aligned_face_mae_8bit_max"]),
        "inverse_aligned_face_p99": face_p99
        <= float(gates["inverse_aligned_face_p99_8bit_max"]),
    }
    return {
        "case_id": case_id,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "height_ratio": ratio,
            "target_height_ratio": float(target["height_ratio"]),
            "height_ratio_tolerance": tolerance,
            "foot_anchor_delta_px": foot_delta,
            "center_x_delta_px": center_delta,
            "limb_gap_area_retention": retained,
            "inverse_aligned_face_mae_8bit": face_mae,
            "inverse_aligned_face_p99_8bit": face_p99,
        },
    }


def _normalized_slice(
    size: tuple[int, int], bounds: tuple[float, float, float, float]
) -> tuple[slice, slice]:
    x0, y0 = _normalized_pixel(bounds[:2], size)
    x1, y1 = _normalized_pixel(bounds[2:], size)
    return slice(y0, y1 + 1), slice(x0, x1 + 1)


def _evaluate_glasses_metrics(
    *,
    source: Image.Image,
    candidate: Image.Image,
    edit_core: Image.Image,
    transition_band: Image.Image,
) -> dict[str, Any]:
    gates = CASE_MASK_CONTRACTS["glasses"]["quantitative_gates"]
    source_values = np.asarray(source.convert("RGB"), dtype=np.int16)
    candidate_values = np.asarray(candidate.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(candidate_values - source_values), axis=2) >= 4
    left_slice = _normalized_slice(source.size, (0.425, 0.125, 0.498, 0.170))
    right_slice = _normalized_slice(source.size, (0.502, 0.125, 0.575, 0.170))
    bridge_slice = _normalized_slice(source.size, (0.496, 0.132, 0.504, 0.158))

    def changed_center_y(region: tuple[slice, slice]) -> tuple[int, float | None]:
        local = changed[region]
        rows, _ = np.where(local)
        if not rows.size:
            return 0, None
        return int(rows.size), float(rows.mean() + int(region[0].start or 0))

    left_count, left_center = changed_center_y(left_slice)
    right_count, right_center = changed_center_y(right_slice)
    bridge_count = int(np.count_nonzero(changed[bridge_slice]))
    center_delta = (
        float(abs(left_center - right_center))
        if left_center is not None and right_center is not None
        else float("inf")
    )
    core = np.asarray(edit_core.convert("L"), dtype=np.uint8) == 255
    band = np.asarray(transition_band.convert("L"), dtype=np.uint8) == 255
    face_slice = _normalized_slice(source.size, (0.38, 0.105, 0.62, 0.225))
    protected_face = changed[face_slice] & ~(core | band)[face_slice]
    protected_count = int(np.count_nonzero(protected_face))
    checks = {
        "bilateral_eye_change": left_count > 0 and right_count > 0,
        "lens_center_y_alignment": center_delta
        <= float(gates["lens_center_y_delta_px_max"]),
        "bridge_occupancy": bridge_count > 0,
        "protected_face": protected_count
        <= int(gates["protected_face_changed_pixels_max"]),
    }
    return {
        "case_id": "glasses",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "left_eye_changed_pixels": left_count,
            "right_eye_changed_pixels": right_count,
            "lens_center_y_delta_px": center_delta,
            "bridge_changed_pixels": bridge_count,
            "protected_face_changed_pixels": protected_count,
        },
    }


def _evaluate_hat_metrics(
    *,
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
) -> dict[str, Any]:
    gates = CASE_MASK_CONTRACTS["hat"]["quantitative_gates"]
    source_values = np.asarray(source.convert("RGB"), dtype=np.int16)
    candidate_values = np.asarray(candidate.convert("RGB"), dtype=np.int16)
    source_a = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    candidate_a = np.asarray(candidate_alpha.convert("L"), dtype=np.uint8)
    source_fg = source_a >= 128
    candidate_fg = candidate_a >= 128
    added = candidate_fg & ~source_fg
    labels, count = ndimage.label(added)
    minimum_component_area = max(4, int(round(added.size * 1e-5)))
    component_areas = [
        int(np.count_nonzero(labels == label)) for label in range(1, count + 1)
    ]
    retained_components = [area for area in component_areas if area >= minimum_component_area]
    contact = bool(
        np.any(ndimage.binary_dilation(added, iterations=2) & source_fg)
    )
    changed = (
        np.max(np.abs(candidate_values - source_values), axis=2) >= 4
    ) | (candidate_a != source_a)
    protected_y = int(
        round(
            float(CASE_MASK_CONTRACTS["hat"]["protected_below_y"])
            * (source.height - 1)
        )
    )
    below_count = int(np.count_nonzero(changed[protected_y:]))
    checks = {
        "added_component_count": len(retained_components)
        == int(gates["added_component_count"]),
        "must_contact_head": contact is bool(gates["must_contact_head"]),
        "changed_pixels_below_brow": below_count
        <= int(gates["changed_pixels_below_brow_max"]),
    }
    return {
        "case_id": "hat",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "added_component_count": len(retained_components),
            "added_component_areas_px": retained_components,
            "minimum_component_area_px": minimum_component_area,
            "must_contact_head": contact,
            "protected_below_y_px": protected_y,
            "changed_pixels_below_brow": below_count,
        },
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size == 0:
        return 0.0
    if np.array_equal(left, right):
        return 1.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _lower_pair_fractions(
    changed: np.ndarray, core: np.ndarray
) -> tuple[float, float]:
    columns = np.where(core)[1]
    if columns.size == 0:
        return 0.0, 0.0
    center = (int(columns.min()) + int(columns.max())) / 2.0
    x = np.arange(core.shape[1])[None, :]
    left = core & (x < center)
    right = core & (x >= center)

    def fraction(region: np.ndarray) -> float:
        count = int(np.count_nonzero(region))
        return float(np.count_nonzero(changed & region) / count) if count else 0.0

    return fraction(left), fraction(right)


def _foot_contact_delta(
    source_alpha: np.ndarray, candidate_alpha: np.ndarray
) -> int:
    width = source_alpha.shape[1]
    deltas = []
    for x0, x1 in ((0, width // 2), (width // 2, width)):
        source_rows = np.where(source_alpha[:, x0:x1] >= 128)[0]
        candidate_rows = np.where(candidate_alpha[:, x0:x1] >= 128)[0]
        if not source_rows.size or not candidate_rows.size:
            return source_alpha.shape[0]
        deltas.append(abs(int(source_rows.max()) - int(candidate_rows.max())))
    return max(deltas)


def _evaluate_color_metrics(
    case_id: str,
    *,
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
    edit_core: Image.Image,
) -> dict[str, Any]:
    contract = CASE_MASK_CONTRACTS[case_id]
    gates = contract["quantitative_gates"]
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    candidate_rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    source_a = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    candidate_a = np.asarray(candidate_alpha.convert("L"), dtype=np.uint8)
    core = np.asarray(edit_core.convert("L"), dtype=np.uint8) == 255
    if not np.any(core):
        raise ValueError("color candidate metric core is empty")
    source_lab = _srgb_to_lab(source_rgb)
    candidate_lab = _srgb_to_lab(candidate_rgb)
    delta_e = np.linalg.norm(candidate_lab - source_lab, axis=2)
    changed = delta_e >= 5.0
    changed_fraction = float(np.count_nonzero(changed & core) / np.count_nonzero(core))
    color_name = str(contract["target_parameters"]["color"])
    target_rgb = np.asarray(TARGET_COLOR_SRGB[color_name], dtype=np.uint8).reshape(1, 1, 3)
    target_lab = _srgb_to_lab(target_rgb)[0, 0]
    median_candidate_lab = np.median(candidate_lab[core], axis=0)
    target_delta_e = float(np.linalg.norm(median_candidate_lab - target_lab))
    source_luminance = (
        source_rgb[..., 0] * 0.2126
        + source_rgb[..., 1] * 0.7152
        + source_rgb[..., 2] * 0.0722
    )
    candidate_luminance = (
        candidate_rgb[..., 0] * 0.2126
        + candidate_rgb[..., 1] * 0.7152
        + candidate_rgb[..., 2] * 0.0722
    )
    luminance_correlation = _safe_correlation(
        source_luminance[core], candidate_luminance[core]
    )
    source_edges = np.hypot(
        ndimage.sobel(source_luminance, axis=0),
        ndimage.sobel(source_luminance, axis=1),
    )
    candidate_edges = np.hypot(
        ndimage.sobel(candidate_luminance, axis=0),
        ndimage.sobel(candidate_luminance, axis=1),
    )
    edge_correlation = _safe_correlation(source_edges[core], candidate_edges[core])
    alpha_identical = bool(np.array_equal(source_a, candidate_a))
    checks: dict[str, bool] = {
        "source_alpha_byte_identical": alpha_identical,
        "target_median_delta_e": target_delta_e
        <= float(gates.get("target_median_delta_e_max", 10.0)),
        "edge_correlation": edge_correlation >= float(gates["edge_correlation_min"]),
    }
    metrics: dict[str, Any] = {
        "source_alpha_byte_identical": alpha_identical,
        "semantic_core_changed_fraction": changed_fraction,
        "target_median_delta_e": target_delta_e,
        "luminance_correlation": luminance_correlation,
        "edge_correlation": edge_correlation,
        "target_color_srgb": list(TARGET_COLOR_SRGB[color_name]),
    }
    if case_id == "short_sleeve_color":
        checks["semantic_core_changed_fraction"] = changed_fraction >= float(
            gates["semantic_core_changed_fraction_min"]
        )
        checks["luminance_correlation"] = luminance_correlation >= float(
            gates["luminance_correlation_min"]
        )
    else:
        left_fraction, right_fraction = _lower_pair_fractions(changed, core)
        key = (
            "each_leg_changed_fraction"
            if case_id == "trousers"
            else "each_shoe_changed_fraction"
        )
        checks[key] = min(left_fraction, right_fraction) >= float(
            gates[f"{key}_min"]
        )
        metrics[f"left_{key}"] = left_fraction
        metrics[f"right_{key}"] = right_fraction
        if case_id == "trousers":
            checks["luminance_correlation"] = luminance_correlation >= float(
                gates["luminance_correlation_min"]
            )
        else:
            contact_delta = _foot_contact_delta(source_a, candidate_a)
            checks["sole_contact_y"] = contact_delta <= int(
                gates["sole_contact_y_tolerance_px"]
            )
            metrics["sole_contact_y_delta_px"] = contact_delta
    return {
        "case_id": case_id,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
    }


def evaluate_candidate_metrics(
    case_id: str,
    *,
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
    edit_core: Image.Image,
    transition_band: Image.Image,
) -> dict[str, Any]:
    """Evaluate the case-specific quantitative gate before visual review."""
    if case_id not in CASE_MASK_CONTRACTS:
        raise ValueError(f"unknown Route-2 attribute case: {case_id}")
    sizes = {
        source.size,
        candidate.size,
        source_alpha.size,
        candidate_alpha.size,
        edit_core.size,
        transition_band.size,
    }
    if len(sizes) != 1:
        raise ValueError("candidate metric inputs differ in canvas size")
    if case_id in {"tall_man", "short_woman"}:
        return _evaluate_height_metrics(
            case_id,
            source=source,
            candidate=candidate,
            source_alpha=source_alpha,
            candidate_alpha=candidate_alpha,
        )
    if case_id == "glasses":
        return _evaluate_glasses_metrics(
            source=source,
            candidate=candidate,
            edit_core=edit_core,
            transition_band=transition_band,
        )
    if case_id == "hat":
        return _evaluate_hat_metrics(
            source=source,
            candidate=candidate,
            source_alpha=source_alpha,
            candidate_alpha=candidate_alpha,
        )
    if case_id in {"short_sleeve_color", "trousers", "shoes"}:
        return _evaluate_color_metrics(
            case_id,
            source=source,
            candidate=candidate,
            source_alpha=source_alpha,
            candidate_alpha=candidate_alpha,
            edit_core=edit_core,
        )
    raise ValueError(f"candidate metrics are not implemented for {case_id}")


def render_mask_overlay(
    source: Image.Image,
    edit_core: Image.Image,
    transition_band: Image.Image,
) -> Image.Image:
    base = source.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    values = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    core = np.asarray(edit_core.convert("L")) == 255
    band = np.asarray(transition_band.convert("L")) == 255
    values[core] = (255, 0, 0, 112)
    values[band] = (255, 196, 0, 112)
    overlay = Image.fromarray(values, "RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_difference(source: Image.Image, candidate: Image.Image) -> Image.Image:
    left = np.asarray(source.convert("RGB"), dtype=np.int16)
    right = np.asarray(candidate.convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("diff images differ in shape")
    difference = np.clip(np.abs(right - left) * 4, 0, 255).astype(np.uint8)
    return Image.fromarray(difference, "RGB")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def publish_mask_bundle(
    *,
    case_id: str,
    source: Image.Image,
    source_alpha: Image.Image,
    source_image_record: Mapping[str, Any],
    source_alpha_record: Mapping[str, Any],
    destination: Path,
) -> Path:
    destination = Path(destination).absolute()
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise ValueError("mask bundle parent must be a direct real directory")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    bundle = construct_mask_bundle(case_id, source, source_alpha)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".staging", dir=parent)
    )
    try:
        images = {
            "edit_core.png": bundle["edit_core"],
            "transition_band.png": bundle["transition_band"],
            "protected_guard.png": bundle["protected_guard"],
            "overlay.png": render_mask_overlay(
                source, bundle["edit_core"], bundle["transition_band"]
            ),
        }
        for filename, image in images.items():
            image.save(staging / filename, format="PNG")
        artifacts = {
            filename: {
                "path": str(destination / filename),
                "sha256": _sha256_file(staging / filename),
                "size_bytes": (staging / filename).stat().st_size,
            }
            for filename in sorted(images)
        }
        payload = {
            "schema": "human_attribute_mask_bundle_v2",
            "case_id": case_id,
            "base_asset_id": bundle["base_asset_id"],
            "construction_version": bundle["construction_version"],
            "strategy": bundle["strategy"],
            "target_parameters": bundle["target_parameters"],
            "quantitative_gates": bundle["quantitative_gates"],
            "source_image": dict(source_image_record),
            "source_alpha": dict(source_alpha_record),
            "metrics": bundle["metrics"],
            "assets": artifacts,
            "agent_visual_review": "pending_agent_mask_overlay_qa",
            "user_acceptance": "pending_user_review",
        }
        if "user_approved" in json.dumps(payload):
            raise ValueError("mask bundle may not claim user approval")
        manifest = staging / "mask_manifest.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in staging.iterdir():
            path.chmod(0o444)
        _rename_noreplace(staging, destination)
        return destination / "mask_manifest.json"
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def mask_agent_decision_path(bundle_dir: Path) -> Path:
    bundle_dir = Path(bundle_dir).absolute()
    return bundle_dir.with_name(f"{bundle_dir.name}.agent_mask_qa.json")


def mask_correction_path(bundle_dir: Path) -> Path:
    bundle_dir = Path(bundle_dir).absolute()
    return bundle_dir.with_name(
        f"{bundle_dir.name}.agent_mask_qa_correction_v1.json"
    )


def _validated_mask_snapshot(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).absolute()
    if bundle_dir.is_symlink() or not bundle_dir.is_dir() or bundle_dir.resolve() != bundle_dir:
        raise ValueError("mask review bundle must be a direct real directory")
    manifest = bundle_dir / "mask_manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.resolve() != manifest:
        raise ValueError("mask review manifest is missing")
    if manifest.stat().st_mode & 0o777 != 0o444:
        raise ValueError("mask review manifest must have mode 0444")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"mask review manifest is unreadable: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "human_attribute_mask_bundle_v2"
        or payload.get("agent_visual_review") != "pending_agent_mask_overlay_qa"
        or payload.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(payload)
    ):
        raise ValueError("mask review manifest state is invalid")
    artifacts = payload.get("assets")
    expected = {
        "edit_core.png",
        "transition_band.png",
        "protected_guard.png",
        "overlay.png",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected:
        raise ValueError("mask review artifact set is incomplete")
    snapshot_artifacts = {}
    for filename in sorted(expected):
        path = bundle_dir / filename
        record = artifacts[filename]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or not isinstance(record, Mapping)
            or record.get("path") != str(path)
            or record.get("sha256") != _sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"mask review artifact snapshot changed: {filename}")
        if path.stat().st_mode & 0o777 != 0o444:
            raise ValueError(f"mask review artifact is mutable: {filename}")
        snapshot_artifacts[filename] = {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    return {
        "case_id": payload["case_id"],
        "manifest_sha256": _sha256_file(manifest),
        "artifacts": snapshot_artifacts,
    }


def record_mask_agent_qa(
    bundle_dir: Path,
    *,
    status: str,
    reviewer: str,
    notes: str,
    checks: Mapping[str, bool],
) -> Path:
    if status not in {MASK_AGENT_PASS, "rejected"}:
        raise ValueError("mask agent QA status is invalid")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("mask agent QA reviewer is empty")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("mask agent QA notes are empty")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(MASK_AGENT_VISUAL_CHECKS)
        or any(not isinstance(value, bool) for value in checks.values())
        or (status == MASK_AGENT_PASS and not all(checks.values()))
    ):
        raise ValueError("mask agent QA checklist is incomplete or failed")
    snapshot = _validated_mask_snapshot(bundle_dir)
    destination = mask_agent_decision_path(bundle_dir)
    payload = {
        "schema": "human_attribute_mask_agent_qa_v1",
        "case_id": snapshot["case_id"],
        "status": status,
        "reviewer_kind": "agent",
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "checks": dict(checks),
        "snapshot": snapshot,
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise ValueError("mask agent QA may not claim user approval")
    parent = destination.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _rename_noreplace(temporary, destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def record_rejected_mask_correction(
    bundle_dir: Path,
    *,
    prior_decision: Path,
    replacement_bundle: Path,
    reviewer: str,
    reason: str,
) -> Path:
    """Append an immutable rejection without altering the historical decision."""
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("mask correction reviewer is empty")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("mask correction reason is empty")
    old_snapshot = _validated_mask_snapshot(bundle_dir)
    prior = Path(prior_decision).absolute()
    expected_prior = mask_agent_decision_path(bundle_dir)
    if (
        prior != expected_prior
        or prior.is_symlink()
        or not prior.is_file()
        or prior.resolve() != prior
        or prior.stat().st_mode & 0o777 != 0o444
    ):
        raise ValueError("prior mask decision is not the immutable canonical decision")
    replacement_snapshot = _validated_mask_snapshot(replacement_bundle)
    if replacement_snapshot["case_id"] != old_snapshot["case_id"]:
        raise ValueError("replacement mask case differs from the rejected snapshot")
    destination = mask_correction_path(bundle_dir)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    payload = {
        "schema": "human_attribute_mask_qa_correction_v1",
        "case_id": old_snapshot["case_id"],
        "status": "rejected",
        "reviewer_kind": "agent",
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "old_snapshot": old_snapshot,
        "prior_decision": {
            "path": str(prior),
            "sha256": _sha256_file(prior),
            "size_bytes": prior.stat().st_size,
        },
        "replacement": replacement_snapshot,
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise ValueError("mask correction may not claim user approval")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _rename_noreplace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def assert_mask_agent_qa_passed(bundle_dir: Path) -> dict[str, Any]:
    destination = mask_agent_decision_path(bundle_dir)
    if destination.is_symlink() or not destination.is_file() or destination.resolve() != destination:
        raise ValueError("mask agent QA decision is pending")
    if destination.stat().st_mode & 0o222:
        raise ValueError("mask agent QA decision is mutable")
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"mask agent QA decision is unreadable: {error}") from error
    try:
        snapshot = _validated_mask_snapshot(bundle_dir)
    except ValueError as error:
        raise ValueError(f"mask review snapshot changed: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "human_attribute_mask_agent_qa_v1"
        or payload.get("case_id") != snapshot["case_id"]
        or payload.get("status") != MASK_AGENT_PASS
        or payload.get("reviewer_kind") != "agent"
        or payload.get("snapshot") != snapshot
        or payload.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(payload)
        or set(payload.get("checks", {})) != set(MASK_AGENT_VISUAL_CHECKS)
        or not all(value is True for value in payload["checks"].values())
    ):
        raise ValueError("mask agent QA decision or snapshot changed")
    return payload
