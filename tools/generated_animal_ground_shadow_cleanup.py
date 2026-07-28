#!/usr/bin/env python3
"""Fail-closed cleanup of one opaque ground-shadow component.

This CPU-only utility is intentionally narrower than a general segmentation
editor.  It authenticates an RGB source candidate, its binary alpha mask and
the derived RGBA input, then removes exactly one color- and space-bounded
8-connected component selected by a reviewed seed box.  Four protected paw
probes must remain byte-exact, and the retained animal must remain one
8-connected component.

The output is technical input-repair evidence only.  It cannot authorize asset
registration or dynamic admission.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


SCHEMA = "avengine_generated_animal_opaque_ground_shadow_cleanup_v2"
OUTPUT_ALPHA_NAME = "alpha_shadow_cleaned.png"
OUTPUT_RGBA_NAME = "input_rgba_shadow_cleaned.png"
OUTPUT_REMOVAL_MASK_NAME = "removed_shadow_mask.png"
OUTPUT_MANIFEST_NAME = "shadow_cleanup_manifest.json"

_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class GroundShadowCleanupError(ValueError):
    """Raised when an input, rule, or cleanup result is not safely bounded."""


@dataclass(frozen=True)
class Box:
    """Half-open image-space box in ``x0, y0, x1, y1`` order."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass(frozen=True)
class PawProbe:
    name: str
    box: Box
    minimum_opaque_pixels: int


@dataclass(frozen=True)
class CleanupRule:
    cleanup_roi: Box
    seed_boxes: tuple[Box, ...]
    rgb_channel_spread_max: int
    rgb_sum_min: int
    rgb_sum_max: int
    protected_light_rgb_sum_min: int
    protected_chroma_spread_min: int
    minimum_selected_component_pixels: int
    maximum_selected_component_pixels: int
    minimum_removed_pixels: int
    maximum_removed_pixels: int
    maximum_removed_foreground_fraction: float
    maximum_satellite_pixels: int
    paw_probes: tuple[PawProbe, ...]
    body_probes: tuple[PawProbe, ...]


@dataclass(frozen=True)
class _Component:
    label: int
    pixel_count: int
    bbox_xyxy_inclusive: tuple[int, int, int, int]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GroundShadowCleanupError(f"{label} must be lowercase SHA-256")
    return value


def _read_authenticated(path: Path, expected_sha256: str, label: str) -> bytes:
    expected = _validated_sha256(expected_sha256, f"{label} expected SHA-256")
    if not path.is_file():
        raise GroundShadowCleanupError(f"{label} is not a file: {path}")
    value = path.read_bytes()
    actual = sha256_bytes(value)
    if actual != expected:
        raise GroundShadowCleanupError(
            f"{label} SHA-256 changed: expected {expected}, got {actual}"
        )
    return value


def _load_png_exact(value: bytes, expected_mode: str, label: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(value)) as image:
            image.load()
            if image.format != "PNG":
                raise GroundShadowCleanupError(f"{label} must be PNG")
            if image.mode != expected_mode:
                raise GroundShadowCleanupError(
                    f"{label} mode changed: expected {expected_mode}, got {image.mode}"
                )
            result = np.array(image)
    except GroundShadowCleanupError:
        raise
    except Exception as error:
        raise GroundShadowCleanupError(f"{label} cannot be decoded: {error}") from error
    return result


def _validate_box(box: Box, width: int, height: int, label: str) -> None:
    coordinates = (box.x0, box.y0, box.x1, box.y1)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in coordinates):
        raise GroundShadowCleanupError(f"{label} coordinates must be integers")
    if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
        raise GroundShadowCleanupError(
            f"{label} is outside {width}x{height}: {box.as_list()}"
        )
    if box.width <= 0 or box.height <= 0:
        raise GroundShadowCleanupError(f"{label} must have positive area")


def _boxes_overlap(first: Box, second: Box) -> bool:
    return not (
        first.x1 <= second.x0
        or second.x1 <= first.x0
        or first.y1 <= second.y0
        or second.y1 <= first.y0
    )


def _validate_rule(rule: CleanupRule, width: int, height: int) -> None:
    _validate_box(rule.cleanup_roi, width, height, "cleanup ROI")
    if not 1 <= len(rule.seed_boxes) <= 4:
        raise GroundShadowCleanupError("one to four seed boxes are required")
    for index, seed_box in enumerate(rule.seed_boxes):
        _validate_box(seed_box, width, height, f"seed box {index}")
        if (
            seed_box.x0 < rule.cleanup_roi.x0
            or seed_box.y0 < rule.cleanup_roi.y0
            or seed_box.x1 > rule.cleanup_roi.x1
            or seed_box.y1 > rule.cleanup_roi.y1
        ):
            raise GroundShadowCleanupError(
                f"seed box {index} must be contained by cleanup ROI"
            )
    if rule.cleanup_roi.area > width * height * 0.25:
        raise GroundShadowCleanupError("cleanup ROI exceeds hard 25% canvas bound")

    integer_fields = {
        "rgb_channel_spread_max": rule.rgb_channel_spread_max,
        "rgb_sum_min": rule.rgb_sum_min,
        "rgb_sum_max": rule.rgb_sum_max,
        "protected_light_rgb_sum_min": rule.protected_light_rgb_sum_min,
        "protected_chroma_spread_min": rule.protected_chroma_spread_min,
        "minimum_selected_component_pixels": rule.minimum_selected_component_pixels,
        "maximum_selected_component_pixels": rule.maximum_selected_component_pixels,
        "minimum_removed_pixels": rule.minimum_removed_pixels,
        "maximum_removed_pixels": rule.maximum_removed_pixels,
        "maximum_satellite_pixels": rule.maximum_satellite_pixels,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in integer_fields.values()
    ):
        raise GroundShadowCleanupError("all integer cleanup gates must be integers")
    if not 0 <= rule.rgb_channel_spread_max <= 64:
        raise GroundShadowCleanupError("RGB channel spread maximum must be in [0, 64]")
    if not 0 <= rule.rgb_sum_min <= rule.rgb_sum_max <= 765:
        raise GroundShadowCleanupError("RGB sum bounds must be ordered inside [0, 765]")
    if not rule.rgb_sum_max < rule.protected_light_rgb_sum_min <= 765:
        raise GroundShadowCleanupError(
            "protected light RGB sum must be above shadow RGB sum maximum"
        )
    if not (
        rule.rgb_channel_spread_max
        < rule.protected_chroma_spread_min
        <= 255
    ):
        raise GroundShadowCleanupError(
            "protected chroma spread must be above shadow channel-spread maximum"
        )
    if not (
        0
        < rule.minimum_selected_component_pixels
        <= rule.maximum_selected_component_pixels
    ):
        raise GroundShadowCleanupError("selected-component pixel gates are invalid")
    if not 0 < rule.minimum_removed_pixels <= rule.maximum_removed_pixels:
        raise GroundShadowCleanupError("removed-pixel gates are invalid")
    if not 0 <= rule.maximum_satellite_pixels <= 1024:
        raise GroundShadowCleanupError("satellite bound must be in [0, 1024]")
    if (
        not isinstance(rule.maximum_removed_foreground_fraction, (int, float))
        or isinstance(rule.maximum_removed_foreground_fraction, bool)
        or not 0.0 < float(rule.maximum_removed_foreground_fraction) <= 0.1
    ):
        raise GroundShadowCleanupError(
            "removed foreground fraction must be in (0, 0.1]"
        )

    if len(rule.paw_probes) != 4:
        raise GroundShadowCleanupError("exactly four paw probes are required")
    if not 1 <= len(rule.body_probes) <= 8:
        raise GroundShadowCleanupError("one to eight body probes are required")
    names: set[str] = set()
    all_probes = rule.paw_probes + rule.body_probes
    for index, probe in enumerate(all_probes):
        if (
            not isinstance(probe.name, str)
            or not probe.name
            or probe.name in names
        ):
            raise GroundShadowCleanupError(
                "protected probe names must be unique and non-empty"
            )
        names.add(probe.name)
        _validate_box(probe.box, width, height, f"protected probe {probe.name}")
        if (
            not isinstance(probe.minimum_opaque_pixels, int)
            or isinstance(probe.minimum_opaque_pixels, bool)
            or probe.minimum_opaque_pixels <= 0
        ):
            raise GroundShadowCleanupError(
                f"protected probe {probe.name} minimum must be positive"
            )
        if probe.minimum_opaque_pixels > probe.box.area:
            raise GroundShadowCleanupError(
                f"protected probe {probe.name} minimum exceeds probe area"
            )
        for previous in all_probes[:index]:
            if _boxes_overlap(probe.box, previous.box):
                raise GroundShadowCleanupError(
                    f"protected probes {probe.name} and {previous.name} overlap"
                )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, tuple[_Component, ...]]:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise GroundShadowCleanupError("component mask must be two-dimensional bool")
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    components: list[_Component] = []
    next_label = 0

    for flat_index in np.flatnonzero(mask):
        y, x = divmod(int(flat_index), width)
        if labels[y, x] != 0:
            continue
        next_label += 1
        labels[y, x] = next_label
        queue: deque[tuple[int, int]] = deque([(y, x)])
        count = 0
        min_x = max_x = x
        min_y = max_y = y
        while queue:
            current_y, current_x = queue.popleft()
            count += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for delta_y, delta_x in _NEIGHBORS_8:
                neighbor_y = current_y + delta_y
                neighbor_x = current_x + delta_x
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and labels[neighbor_y, neighbor_x] == 0
                ):
                    labels[neighbor_y, neighbor_x] = next_label
                    queue.append((neighbor_y, neighbor_x))
        components.append(
            _Component(
                label=next_label,
                pixel_count=count,
                bbox_xyxy_inclusive=(min_x, min_y, max_x, max_y),
            )
        )
    return labels, tuple(components)


def _dilate_8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for delta_y in range(3):
        for delta_x in range(3):
            result |= padded[
                delta_y : delta_y + height,
                delta_x : delta_x + width,
            ]
    return result


def _bbox(mask: np.ndarray, label: str) -> list[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise GroundShadowCleanupError(f"{label} is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_sha256(mask: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())


def _probe_evidence(
    alpha_before: np.ndarray,
    alpha_after: np.ndarray,
    probe: PawProbe,
) -> dict[str, Any]:
    slices = probe.box.slices()
    before = alpha_before[slices]
    after = alpha_after[slices]
    count = int(before.sum())
    if count < probe.minimum_opaque_pixels:
        raise GroundShadowCleanupError(
            f"paw probe {probe.name} has only {count} opaque pixels; "
            f"minimum is {probe.minimum_opaque_pixels}"
        )
    if not np.array_equal(before, after):
        changed = int(np.count_nonzero(before != after))
        raise GroundShadowCleanupError(
            f"paw probe {probe.name} changed at {changed} pixels"
        )
    local_y, local_x = np.where(before)
    return {
        "name": probe.name,
        "box_xyxy_half_open": probe.box.as_list(),
        "minimum_opaque_pixels": probe.minimum_opaque_pixels,
        "opaque_pixel_count_before": count,
        "opaque_pixel_count_after": int(after.sum()),
        "alpha_mask_sha256_before": _mask_sha256(before),
        "alpha_mask_sha256_after": _mask_sha256(after),
        "opaque_bbox_xyxy_inclusive": [
            int(local_x.min()) + probe.box.x0,
            int(local_y.min()) + probe.box.y0,
            int(local_x.max()) + probe.box.x0,
            int(local_y.max()) + probe.box.y0,
        ],
        "bottommost_opaque_y_before": int(local_y.max()) + probe.box.y0,
        "bottommost_opaque_y_after": int(local_y.max()) + probe.box.y0,
        "byte_exact_preserved": True,
    }


def _encode_png(array: np.ndarray, mode: str) -> bytes:
    output = BytesIO()
    Image.fromarray(array, mode=mode).save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


def _output_record(path: Path, value: bytes, mode: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_bytes(value),
        "size_bytes": len(value),
        "mode": mode,
    }


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as output:
        output.write(value)


def clean_authenticated_ground_shadow(
    *,
    source_candidate_path: Path,
    expected_source_candidate_sha256: str,
    input_alpha_path: Path,
    expected_input_alpha_sha256: str,
    input_rgba_path: Path,
    expected_input_rgba_sha256: str,
    output_dir: Path,
    pixal_seed: int,
    rule: CleanupRule,
) -> dict[str, Any]:
    """Run one authenticated, bounded CPU cleanup and write immutable evidence."""

    if not isinstance(pixal_seed, int) or isinstance(pixal_seed, bool) or pixal_seed < 0:
        raise GroundShadowCleanupError("Pixal seed must be a non-negative integer")

    candidate_bytes = _read_authenticated(
        source_candidate_path,
        expected_source_candidate_sha256,
        "source candidate",
    )
    alpha_bytes = _read_authenticated(
        input_alpha_path,
        expected_input_alpha_sha256,
        "input alpha",
    )
    rgba_bytes = _read_authenticated(
        input_rgba_path,
        expected_input_rgba_sha256,
        "input RGBA",
    )
    candidate_rgb = _load_png_exact(candidate_bytes, "RGB", "source candidate")
    input_alpha_u8 = _load_png_exact(alpha_bytes, "L", "input alpha")
    input_rgba = _load_png_exact(rgba_bytes, "RGBA", "input RGBA")

    if candidate_rgb.shape[:2] != input_alpha_u8.shape:
        raise GroundShadowCleanupError("candidate and alpha dimensions changed")
    if candidate_rgb.shape[:2] != input_rgba.shape[:2]:
        raise GroundShadowCleanupError("candidate and RGBA dimensions changed")
    if not np.array_equal(input_rgba[:, :, :3], candidate_rgb):
        raise GroundShadowCleanupError("input RGBA RGB is not byte-exact candidate RGB")
    if not np.array_equal(input_rgba[:, :, 3], input_alpha_u8):
        raise GroundShadowCleanupError("input RGBA alpha is not byte-exact alpha PNG")

    alpha_values = np.unique(input_alpha_u8).tolist()
    if alpha_values != [0, 255]:
        raise GroundShadowCleanupError(
            f"input alpha must be strictly binary [0, 255], got {alpha_values}"
        )
    alpha_before = input_alpha_u8 == 255
    height, width = alpha_before.shape
    _validate_rule(rule, width, height)

    input_labels, input_components = _label_components(alpha_before)
    del input_labels
    if len(input_components) != 1:
        raise GroundShadowCleanupError(
            f"input foreground must be one 8-connected component, got "
            f"{len(input_components)}"
        )
    foreground_before = int(alpha_before.sum())

    y_grid, x_grid = np.indices(alpha_before.shape)
    roi_mask = (
        (x_grid >= rule.cleanup_roi.x0)
        & (x_grid < rule.cleanup_roi.x1)
        & (y_grid >= rule.cleanup_roi.y0)
        & (y_grid < rule.cleanup_roi.y1)
    )
    rgb_i16 = input_rgba[:, :, :3].astype(np.int16)
    rgb_sum = rgb_i16.sum(axis=2)
    channel_spread = rgb_i16.max(axis=2) - rgb_i16.min(axis=2)
    color_candidate = (
        alpha_before
        & roi_mask
        & (channel_spread <= rule.rgb_channel_spread_max)
        & (rgb_sum >= rule.rgb_sum_min)
        & (rgb_sum <= rule.rgb_sum_max)
    )
    candidate_labels, candidate_components = _label_components(color_candidate)
    selected_labels: list[int] = []
    for index, seed_box in enumerate(rule.seed_boxes):
        seed_labels = sorted(
            int(value)
            for value in np.unique(candidate_labels[seed_box.slices()])
            if value != 0
        )
        if len(seed_labels) != 1:
            raise GroundShadowCleanupError(
                f"seed box {index} must touch exactly one color component, "
                f"got {seed_labels}"
            )
        selected_label = seed_labels[0]
        if selected_label in selected_labels:
            raise GroundShadowCleanupError(
                f"seed box {index} repeats selected component {selected_label}"
            )
        selected_labels.append(selected_label)
    selected_components = [
        next(
            component
            for component in candidate_components
            if component.label == selected_label
        )
        for selected_label in selected_labels
    ]
    for index, selected_component in enumerate(selected_components):
        if not (
            rule.minimum_selected_component_pixels
            <= selected_component.pixel_count
            <= rule.maximum_selected_component_pixels
        ):
            raise GroundShadowCleanupError(
                f"selected shadow component {index} pixel count "
                f"{selected_component.pixel_count} is outside "
                f"[{rule.minimum_selected_component_pixels}, "
                f"{rule.maximum_selected_component_pixels}]"
            )
    selected_mask = np.isin(candidate_labels, selected_labels)
    intermediate = alpha_before & ~selected_mask

    retained_labels, retained_components = _label_components(intermediate)
    if not retained_components:
        raise GroundShadowCleanupError("cleanup removed the complete foreground")
    sizes = [component.pixel_count for component in retained_components]
    largest_size = max(sizes)
    if sizes.count(largest_size) != 1:
        raise GroundShadowCleanupError("retained foreground has no unique primary component")
    primary = next(
        component
        for component in retained_components
        if component.pixel_count == largest_size
    )
    alpha_after = retained_labels == primary.label
    satellites = intermediate & ~alpha_after
    satellite_pixels = int(satellites.sum())
    if satellite_pixels > rule.maximum_satellite_pixels:
        raise GroundShadowCleanupError(
            f"satellite cleanup needs {satellite_pixels} pixels; maximum is "
            f"{rule.maximum_satellite_pixels}"
        )
    if np.any(satellites & ~roi_mask):
        raise GroundShadowCleanupError("satellite cleanup escaped reviewed ROI")
    selected_dilation = _dilate_8(selected_mask)
    for component in retained_components:
        if component.label == primary.label:
            continue
        component_mask = retained_labels == component.label
        if not np.any(component_mask & selected_dilation):
            raise GroundShadowCleanupError(
                "a retained satellite does not touch the selected shadow component"
            )

    removed = alpha_before & ~alpha_after
    removed_pixels = int(removed.sum())
    removed_fraction = removed_pixels / foreground_before
    if not rule.minimum_removed_pixels <= removed_pixels <= rule.maximum_removed_pixels:
        raise GroundShadowCleanupError(
            f"removed pixel count {removed_pixels} is outside "
            f"[{rule.minimum_removed_pixels}, {rule.maximum_removed_pixels}]"
        )
    if removed_fraction > rule.maximum_removed_foreground_fraction:
        raise GroundShadowCleanupError(
            f"removed foreground fraction {removed_fraction:.8f} exceeds "
            f"{rule.maximum_removed_foreground_fraction:.8f}"
        )
    if np.any(removed & ~roi_mask):
        raise GroundShadowCleanupError("cleanup changed alpha outside reviewed ROI")
    protected_light = alpha_before & (
        rgb_sum >= rule.protected_light_rgb_sum_min
    )
    protected_chroma = alpha_before & (
        channel_spread >= rule.protected_chroma_spread_min
    )
    protected_content = protected_light | protected_chroma
    protected_content_removed = int((removed & protected_content).sum())
    if protected_content_removed:
        raise GroundShadowCleanupError(
            "cleanup removed "
            f"{protected_content_removed} protected light/high-chroma pixels"
        )

    bbox_before = _bbox(alpha_before, "input foreground")
    bbox_after = _bbox(alpha_after, "output foreground")
    if bbox_after != bbox_before:
        raise GroundShadowCleanupError(
            f"subject bbox changed: before={bbox_before}, after={bbox_after}"
        )
    final_labels, final_components = _label_components(alpha_after)
    del final_labels
    if len(final_components) != 1:
        raise GroundShadowCleanupError(
            f"output foreground must remain one component, got {len(final_components)}"
        )

    paw_evidence = [
        _probe_evidence(alpha_before, alpha_after, probe)
        for probe in rule.paw_probes
    ]
    body_evidence = [
        _probe_evidence(alpha_before, alpha_after, probe)
        for probe in rule.body_probes
    ]

    visible_before = np.where(
        alpha_before[:, :, None],
        input_rgba[:, :, :3],
        np.uint8(0),
    )
    visible_after = np.where(
        alpha_after[:, :, None],
        input_rgba[:, :, :3],
        np.uint8(0),
    )
    output_alpha = (alpha_after.astype(np.uint8) * 255)
    output_rgba = np.zeros_like(input_rgba)
    output_rgba[:, :, :3] = visible_after
    output_rgba[:, :, 3] = output_alpha
    removal_u8 = removed.astype(np.uint8) * 255

    alpha_output_bytes = _encode_png(output_alpha, "L")
    rgba_output_bytes = _encode_png(output_rgba, "RGBA")
    removal_output_bytes = _encode_png(removal_u8, "L")
    alpha_output_path = output_dir / OUTPUT_ALPHA_NAME
    rgba_output_path = output_dir / OUTPUT_RGBA_NAME
    removal_output_path = output_dir / OUTPUT_REMOVAL_MASK_NAME
    manifest_path = output_dir / OUTPUT_MANIFEST_NAME

    hidden_rgb_nonzero = (~alpha_before) & np.any(input_rgba[:, :, :3] != 0, axis=2)
    rgba_changed = np.any(input_rgba != output_rgba, axis=2)
    visible_changed = np.any(visible_before != visible_after, axis=2)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_classification": "technical_spike_input_repair_only",
        "formal_dataset_registration_authorized": False,
        "execution": {
            "class": "deterministic_cpu_only",
            "gpu_or_pixal_started": False,
            "coordinate_convention": "xyxy_half_open",
            "connectivity": 8,
        },
        "source": {
            "candidate": {
                "path": str(source_candidate_path),
                "sha256": sha256_bytes(candidate_bytes),
                "size_bytes": len(candidate_bytes),
                "mode": "RGB",
                "size_xy": [width, height],
            },
            "input_alpha": {
                "path": str(input_alpha_path),
                "sha256": sha256_bytes(alpha_bytes),
                "size_bytes": len(alpha_bytes),
                "mode": "L",
                "unique_values": alpha_values,
            },
            "input_rgba": {
                "path": str(input_rgba_path),
                "sha256": sha256_bytes(rgba_bytes),
                "size_bytes": len(rgba_bytes),
                "mode": "RGBA",
                "rgb_is_byte_exact_candidate_rgb": True,
                "alpha_is_byte_exact_input_alpha": True,
            },
            # Decimal text is deliberate: the current seed exceeds JavaScript's
            # exact integer range and must survive JSON tooling byte-exactly.
            "pixal_seed_decimal_frozen_for_controlled_rerun": str(pixal_seed),
        },
        "rule": {
            "cleanup_roi_xyxy_half_open": rule.cleanup_roi.as_list(),
            "seed_boxes_xyxy_half_open": [
                seed_box.as_list() for seed_box in rule.seed_boxes
            ],
            "color": {
                "rgb_channel_spread_max": rule.rgb_channel_spread_max,
                "rgb_sum_min": rule.rgb_sum_min,
                "rgb_sum_max": rule.rgb_sum_max,
                "protected_light_rgb_sum_min": (
                    rule.protected_light_rgb_sum_min
                ),
                "protected_chroma_spread_min": (
                    rule.protected_chroma_spread_min
                ),
            },
            "gates": {
                "selected_component_pixels_each": [
                    rule.minimum_selected_component_pixels,
                    rule.maximum_selected_component_pixels,
                ],
                "removed_pixels": [
                    rule.minimum_removed_pixels,
                    rule.maximum_removed_pixels,
                ],
                "maximum_removed_foreground_fraction": (
                    rule.maximum_removed_foreground_fraction
                ),
                "maximum_satellite_pixels": rule.maximum_satellite_pixels,
                "four_paw_probes_must_be_byte_exact": True,
                "body_probes_must_be_byte_exact": True,
                "all_light_and_high_chroma_foreground_must_be_preserved": True,
                "subject_bbox_must_be_exact": True,
                "output_foreground_component_count_must_equal": 1,
            },
            "paw_probes": [
                {
                    "name": probe.name,
                    "box_xyxy_half_open": probe.box.as_list(),
                    "minimum_opaque_pixels": probe.minimum_opaque_pixels,
                }
                for probe in rule.paw_probes
            ],
            "body_probes": [
                {
                    "name": probe.name,
                    "box_xyxy_half_open": probe.box.as_list(),
                    "minimum_opaque_pixels": probe.minimum_opaque_pixels,
                }
                for probe in rule.body_probes
            ],
        },
        "measurements": {
            "foreground_pixel_count_before": foreground_before,
            "foreground_pixel_count_after": int(alpha_after.sum()),
            "input_foreground_component_count": len(input_components),
            "output_foreground_component_count": len(final_components),
            "subject_bbox_xyxy_inclusive_before": bbox_before,
            "subject_bbox_xyxy_inclusive_after": bbox_after,
            "selected_shadow_components": [
                {
                    "seed_box_xyxy_half_open": rule.seed_boxes[index].as_list(),
                    "pixel_count": selected_component.pixel_count,
                    "bbox_xyxy_inclusive": list(
                        selected_component.bbox_xyxy_inclusive
                    ),
                }
                for index, selected_component in enumerate(selected_components)
            ],
            "selected_shadow_pixel_count": int(selected_mask.sum()),
            "satellite_pixel_count": satellite_pixels,
            "removed_alpha_pixel_count": removed_pixels,
            "removed_foreground_fraction": round(removed_fraction, 8),
            "protected_light_pixel_count": int(protected_light.sum()),
            "protected_high_chroma_pixel_count": int(protected_chroma.sum()),
            "protected_light_or_high_chroma_removed_pixel_count": (
                protected_content_removed
            ),
            "changed_visible_rgb_pixel_count": int(visible_changed.sum()),
            "zeroed_preexisting_hidden_rgb_pixel_count": int(hidden_rgb_nonzero.sum()),
            "changed_rgba_pixel_count": int(rgba_changed.sum()),
            "visible_rgb_raw_sha256_before": sha256_bytes(
                np.ascontiguousarray(visible_before).tobytes()
            ),
            "visible_rgb_raw_sha256_after": sha256_bytes(
                np.ascontiguousarray(visible_after).tobytes()
            ),
            "removed_alpha_mask_raw_sha256": _mask_sha256(removed),
            "paw_bottom_evidence": paw_evidence,
            "protected_body_evidence": body_evidence,
        },
        "outputs": {
            "cleaned_alpha": _output_record(
                alpha_output_path, alpha_output_bytes, "L"
            ),
            "cleaned_rgba": _output_record(
                rgba_output_path, rgba_output_bytes, "RGBA"
            ),
            "removed_shadow_mask": _output_record(
                removal_output_path, removal_output_bytes, "L"
            ),
        },
        "decision": {
            "opaque_ground_shadow_cleanup_passed": True,
            "source_candidate_unchanged": True,
            "pixal_seed_unchanged_for_next_run": True,
            "paw_bottom_evidence_preserved": True,
            "abdomen_and_leg_evidence_preserved": True,
            "dynamic_asset_admission_authorized": False,
            "next_step": (
                "visual_review_then_one_same_source_same_seed_pixal_rerun; "
                "four independent limbs and clean underside remain fail-closed gates"
            ),
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    if output_dir.exists():
        raise GroundShadowCleanupError(f"output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise GroundShadowCleanupError(
            f"output parent directory does not exist: {output_dir.parent}"
        )
    output_dir.mkdir()
    _write_exclusive(alpha_output_path, alpha_output_bytes)
    _write_exclusive(rgba_output_path, rgba_output_bytes)
    _write_exclusive(removal_output_path, removal_output_bytes)
    _write_exclusive(manifest_path, manifest_bytes)
    return manifest


def _parse_box(value: str, label: str) -> Box:
    try:
        parts = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must contain four comma-separated integers"
        ) from error
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"{label} must contain four comma-separated integers"
        )
    return Box(*parts)


def _box_argument(value: str) -> Box:
    return _parse_box(value, "box")


def _paw_probe_argument(value: str) -> PawProbe:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "paw probe must be NAME:X0,Y0,X1,Y1:MIN_OPAQUE_PIXELS"
        )
    name, raw_box, raw_minimum = parts
    try:
        minimum = int(raw_minimum)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "paw probe minimum must be an integer"
        ) from error
    return PawProbe(
        name=name,
        box=_parse_box(raw_box, f"paw probe {name} box"),
        minimum_opaque_pixels=minimum,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate", type=Path, required=True)
    parser.add_argument("--expected-source-candidate-sha256", required=True)
    parser.add_argument("--input-alpha", type=Path, required=True)
    parser.add_argument("--expected-input-alpha-sha256", required=True)
    parser.add_argument("--input-rgba", type=Path, required=True)
    parser.add_argument("--expected-input-rgba-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixal-seed", type=int, required=True)
    parser.add_argument("--cleanup-roi", type=_box_argument, required=True)
    parser.add_argument(
        "--seed-box",
        action="append",
        type=_box_argument,
        required=True,
        help="repeat once per reviewed shadow component",
    )
    parser.add_argument("--rgb-channel-spread-max", type=int, required=True)
    parser.add_argument("--rgb-sum-min", type=int, required=True)
    parser.add_argument("--rgb-sum-max", type=int, required=True)
    parser.add_argument("--protected-light-rgb-sum-min", type=int, required=True)
    parser.add_argument("--protected-chroma-spread-min", type=int, required=True)
    parser.add_argument("--minimum-selected-component-pixels", type=int, required=True)
    parser.add_argument("--maximum-selected-component-pixels", type=int, required=True)
    parser.add_argument("--minimum-removed-pixels", type=int, required=True)
    parser.add_argument("--maximum-removed-pixels", type=int, required=True)
    parser.add_argument(
        "--maximum-removed-foreground-fraction", type=float, required=True
    )
    parser.add_argument("--maximum-satellite-pixels", type=int, required=True)
    parser.add_argument(
        "--paw-probe",
        action="append",
        type=_paw_probe_argument,
        required=True,
        help="repeat exactly four times: NAME:X0,Y0,X1,Y1:MIN_OPAQUE_PIXELS",
    )
    parser.add_argument(
        "--body-probe",
        action="append",
        type=_paw_probe_argument,
        required=True,
        help="repeat for protected abdomen or leg evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rule = CleanupRule(
        cleanup_roi=args.cleanup_roi,
        seed_boxes=tuple(args.seed_box),
        rgb_channel_spread_max=args.rgb_channel_spread_max,
        rgb_sum_min=args.rgb_sum_min,
        rgb_sum_max=args.rgb_sum_max,
        protected_light_rgb_sum_min=args.protected_light_rgb_sum_min,
        protected_chroma_spread_min=args.protected_chroma_spread_min,
        minimum_selected_component_pixels=args.minimum_selected_component_pixels,
        maximum_selected_component_pixels=args.maximum_selected_component_pixels,
        minimum_removed_pixels=args.minimum_removed_pixels,
        maximum_removed_pixels=args.maximum_removed_pixels,
        maximum_removed_foreground_fraction=(
            args.maximum_removed_foreground_fraction
        ),
        maximum_satellite_pixels=args.maximum_satellite_pixels,
        paw_probes=tuple(args.paw_probe),
        body_probes=tuple(args.body_probe),
    )
    manifest = clean_authenticated_ground_shadow(
        source_candidate_path=args.source_candidate,
        expected_source_candidate_sha256=args.expected_source_candidate_sha256,
        input_alpha_path=args.input_alpha,
        expected_input_alpha_sha256=args.expected_input_alpha_sha256,
        input_rgba_path=args.input_rgba,
        expected_input_rgba_sha256=args.expected_input_rgba_sha256,
        output_dir=args.output_dir,
        pixal_seed=args.pixal_seed,
        rule=rule,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / OUTPUT_MANIFEST_NAME),
                "cleaned_rgba": manifest["outputs"]["cleaned_rgba"],
                "cleaned_alpha": manifest["outputs"]["cleaned_alpha"],
                "removed_alpha_pixel_count": manifest["measurements"][
                    "removed_alpha_pixel_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
