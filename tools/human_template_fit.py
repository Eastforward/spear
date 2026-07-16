"""Pure NumPy helpers for fitting and texturing a stable human template."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from tools.human_part_transfer import HumanRegion


def _xyz_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return array


def clamp_xy_displacements(
    displacements: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Return XY-only displacements radially clamped to ``max_distance``."""
    if not np.isfinite(max_distance) or max_distance < 0.0:
        raise ValueError("max_distance must be a finite non-negative value")

    out = _xyz_array(displacements, "displacements").copy()
    out[:, 2] = 0.0
    distances = np.linalg.norm(out[:, :2], axis=1)
    clamped = distances > float(max_distance)
    if np.any(clamped):
        out[clamped, :2] *= (float(max_distance) / distances[clamped])[:, None]
    return out


def smooth_xy_displacements(
    displacements: np.ndarray,
    adjacency: Sequence[Iterable[int]],
    fixed_mask: np.ndarray,
    iterations: int,
    blend: float,
) -> np.ndarray:
    """Smooth XY displacements synchronously while fixed vertices stay at zero."""
    out = _xyz_array(displacements, "displacements").copy()
    neighbors_by_vertex = list(adjacency)
    if len(neighbors_by_vertex) != len(out):
        raise ValueError("adjacency must have one entry per displacement")

    fixed = np.asarray(fixed_mask, dtype=bool)
    if fixed.shape != (len(out),):
        raise ValueError("fixed_mask must have one entry per displacement")
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if not np.isfinite(blend) or not 0.0 <= float(blend) <= 1.0:
        raise ValueError("blend must be between zero and one")

    normalized_adjacency: list[np.ndarray] = []
    for neighbors in neighbors_by_vertex:
        indices = np.asarray(tuple(neighbors), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("each adjacency entry must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= len(out)):
            raise ValueError("adjacency contains an out-of-range vertex")
        normalized_adjacency.append(indices)

    out[:, 2] = 0.0
    out[fixed] = 0.0
    for _ in range(int(iterations)):
        previous = out.copy()
        for vertex, neighbors in enumerate(normalized_adjacency):
            if fixed[vertex] or len(neighbors) == 0:
                continue
            neighbor_mean = previous[neighbors, :2].mean(axis=0)
            out[vertex, :2] = (
                (1.0 - float(blend)) * previous[vertex, :2]
                + float(blend) * neighbor_mean
            )
        out[:, 2] = 0.0
        out[fixed] = 0.0
    return out


def triangle_barycentric_3d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return barycentric weights of a point projected onto a 3D triangle."""
    point_array = np.asarray(point, dtype=np.float64)
    triangle_array = np.asarray(triangle, dtype=np.float64)
    if point_array.shape != (3,):
        raise ValueError("point must have shape (3,)")
    if triangle_array.shape != (3, 3):
        raise ValueError("triangle must have shape (3, 3)")

    edge_0 = triangle_array[1] - triangle_array[0]
    edge_1 = triangle_array[2] - triangle_array[0]
    relative = point_array - triangle_array[0]
    dot_00 = float(np.dot(edge_0, edge_0))
    dot_01 = float(np.dot(edge_0, edge_1))
    dot_11 = float(np.dot(edge_1, edge_1))
    dot_20 = float(np.dot(relative, edge_0))
    dot_21 = float(np.dot(relative, edge_1))
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    scale = dot_00 * dot_11
    if scale == 0.0 or abs(denominator) <= np.finfo(np.float64).eps * scale * 16.0:
        raise ValueError("triangle is degenerate")

    weight_1 = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    weight_2 = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    weight_0 = 1.0 - weight_1 - weight_2
    return np.array((weight_0, weight_1, weight_2), dtype=np.float64)


def sample_texture_bilinear(image: np.ndarray, uv: Sequence[float]) -> np.ndarray:
    """Sample an image bilinearly using Blender's bottom-left UV convention."""
    pixels = np.asarray(image)
    if pixels.ndim < 2 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise ValueError("image must have non-empty height and width dimensions")

    uv_array = np.asarray(uv, dtype=np.float64)
    if uv_array.shape != (2,) or not np.isfinite(uv_array).all():
        raise ValueError("uv must contain two finite values")
    u, v = np.clip(uv_array, 0.0, 1.0)
    x = float(u) * (pixels.shape[1] - 1)
    y = (1.0 - float(v)) * (pixels.shape[0] - 1)
    x_0 = int(np.floor(x))
    y_0 = int(np.floor(y))
    x_1 = min(x_0 + 1, pixels.shape[1] - 1)
    y_1 = min(y_0 + 1, pixels.shape[0] - 1)
    x_fraction = x - x_0
    y_fraction = y - y_0

    top = (
        np.asarray(pixels[y_0, x_0], dtype=np.float64) * (1.0 - x_fraction)
        + np.asarray(pixels[y_0, x_1], dtype=np.float64) * x_fraction
    )
    bottom = (
        np.asarray(pixels[y_1, x_0], dtype=np.float64) * (1.0 - x_fraction)
        + np.asarray(pixels[y_1, x_1], dtype=np.float64) * x_fraction
    )
    return top * (1.0 - y_fraction) + bottom * y_fraction


def _matching_regions(
    target_region: HumanRegion | int | None,
    source_region: HumanRegion | int | None,
) -> None:
    if target_region is None and source_region is None:
        return
    if target_region is None or source_region is None:
        raise ValueError("target_region and source_region must be provided together")
    target = HumanRegion(int(target_region))
    source = HumanRegion(int(source_region))
    if target != source:
        raise ValueError(
            f"incompatible human regions: target={target.name.lower()} "
            f"source={source.name.lower()}"
        )


def rasterize_uv_triangle(
    image: np.ndarray,
    mask: np.ndarray,
    target_uv: np.ndarray,
    source_uv: np.ndarray,
    source_image: np.ndarray,
    *,
    target_region: HumanRegion | int | None = None,
    source_region: HumanRegion | int | None = None,
) -> int:
    """Paint one target UV triangle from an exactly matching source region."""
    _matching_regions(target_region, source_region)
    output = np.asarray(image)
    painted_mask = np.asarray(mask)
    source_pixels = np.asarray(source_image)
    if output.ndim < 2 or output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError("image must have non-empty height and width dimensions")
    if painted_mask.shape != output.shape[:2]:
        raise ValueError("mask must match the image height and width")
    if (
        source_pixels.ndim < 2
        or source_pixels.shape[0] == 0
        or source_pixels.shape[1] == 0
    ):
        raise ValueError("source_image must have non-empty height and width dimensions")

    target = np.asarray(target_uv, dtype=np.float64)
    source = np.asarray(source_uv, dtype=np.float64)
    if target.shape != (3, 2) or not np.isfinite(target).all():
        raise ValueError("target_uv must have shape (3, 2) with finite values")
    if source.shape != (3, 2) or not np.isfinite(source).all():
        raise ValueError("source_uv must have shape (3, 2) with finite values")

    height, width = output.shape[:2]
    target_pixels = np.empty((3, 3), dtype=np.float64)
    target_pixels[:, 0] = target[:, 0] * (width - 1)
    target_pixels[:, 1] = (1.0 - target[:, 1]) * (height - 1)
    target_pixels[:, 2] = 0.0
    try:
        triangle_barycentric_3d(target_pixels[0], target_pixels)
    except ValueError:
        return 0

    x_min = max(0, int(np.floor(target_pixels[:, 0].min())))
    x_max = min(width - 1, int(np.ceil(target_pixels[:, 0].max())))
    y_min = max(0, int(np.floor(target_pixels[:, 1].min())))
    y_max = min(height - 1, int(np.ceil(target_pixels[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return 0

    painted = 0
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            point = np.array((x + 0.5, y + 0.5, 0.0), dtype=np.float64)
            weights = triangle_barycentric_3d(point, target_pixels)
            if np.any(weights < -1e-9):
                continue
            sampled = sample_texture_bilinear(source_pixels, weights @ source)
            try:
                output[y, x] = sampled
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "image and source_image texels have incompatible shapes"
                ) from error
            painted_mask[y, x] = True
            painted += 1
    return painted


def dilate_unpainted(
    image: np.ndarray,
    mask: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Grow painted texels into four-neighbor gaps, updating image and mask."""
    output = np.asarray(image)
    painted_mask = np.asarray(mask)
    if output.ndim < 2 or output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError("image must have non-empty height and width dimensions")
    if painted_mask.shape != output.shape[:2]:
        raise ValueError("mask must match the image height and width")
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")

    for _ in range(int(iterations)):
        current_image = output.copy()
        current_mask = painted_mask.astype(bool, copy=True)
        value_sum = np.zeros_like(current_image, dtype=np.float64)
        neighbor_count = np.zeros(current_mask.shape, dtype=np.int64)

        for direction in ("up", "down", "left", "right"):
            shifted_image = np.zeros_like(current_image)
            shifted_mask = np.zeros_like(current_mask)
            if direction == "up":
                shifted_image[1:] = current_image[:-1]
                shifted_mask[1:] = current_mask[:-1]
            elif direction == "down":
                shifted_image[:-1] = current_image[1:]
                shifted_mask[:-1] = current_mask[1:]
            elif direction == "left":
                shifted_image[:, 1:] = current_image[:, :-1]
                shifted_mask[:, 1:] = current_mask[:, :-1]
            else:
                shifted_image[:, :-1] = current_image[:, 1:]
                shifted_mask[:, :-1] = current_mask[:, 1:]

            candidates = shifted_mask & ~current_mask
            expanded = candidates[(...,) + (None,) * (output.ndim - 2)]
            value_sum += np.asarray(shifted_image, dtype=np.float64) * expanded
            neighbor_count += candidates

        fill = neighbor_count > 0
        if not np.any(fill):
            break
        if output.ndim == 2:
            output[fill] = value_sum[fill] / neighbor_count[fill]
        else:
            denominator_shape = (-1,) + (1,) * (output.ndim - 2)
            output[fill] = value_sum[fill] / neighbor_count[fill].reshape(
                denominator_shape
            )
        painted_mask[fill] = True
    return output


def region_color_medians(
    colors: np.ndarray,
    regions: np.ndarray,
    expected_regions: Iterable[int],
) -> dict[int, np.ndarray]:
    """Return a robust median color for every explicitly required region."""
    values = np.asarray(colors, dtype=np.float64)
    labels = np.asarray(regions, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("colors must have shape (N, C)")
    if labels.shape != (len(values),):
        raise ValueError("regions must have one label per color")
    if not np.isfinite(values).all():
        raise ValueError("colors must be finite")
    palette: dict[int, np.ndarray] = {}
    for region in expected_regions:
        key = int(region)
        selected = values[labels == key]
        if len(selected) == 0:
            raise ValueError(f"missing region {key}")
        palette[key] = np.median(selected, axis=0)
    return palette


def region_palette_from_uv_samples(
    source_image: np.ndarray,
    source_uvs: np.ndarray,
    template_regions: np.ndarray,
    expected_regions: Iterable[int],
) -> dict[int, np.ndarray]:
    """Sample locked source UVs, then aggregate them by template region."""
    image = np.asarray(source_image, dtype=np.float64)
    uvs = np.asarray(source_uvs, dtype=np.float64)
    regions = np.asarray(template_regions, dtype=np.int64)
    if image.ndim != 3 or image.shape[2] == 0:
        raise ValueError("source_image must have shape (H, W, C)")
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("source_uvs must have shape (N, 2)")
    if regions.shape != (len(uvs),):
        raise ValueError("template_regions must have one label per source UV")
    samples = np.asarray(
        [sample_texture_bilinear(image, uv) for uv in uvs],
        dtype=np.float64,
    )
    return region_color_medians(samples, regions, expected_regions)


def regularize_region_labels_by_island(
    labels: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Use the dominant semantic label within each four-connected UV island."""
    region_labels = np.asarray(labels, dtype=np.int64)
    painted = np.asarray(mask, dtype=bool)
    if region_labels.ndim != 2 or painted.shape != region_labels.shape:
        raise ValueError("labels and mask must be matching two-dimensional arrays")
    if np.any(region_labels[painted] < 0):
        raise ValueError("painted pixels must have non-negative region labels")

    output = region_labels.copy()
    output[~painted] = -1
    visited = np.zeros_like(painted)
    height, width = painted.shape
    for start_row, start_column in np.argwhere(painted):
        start_row = int(start_row)
        start_column = int(start_column)
        if visited[start_row, start_column]:
            continue
        pending = deque(((start_row, start_column),))
        visited[start_row, start_column] = True
        component = []
        while pending:
            row, column = pending.popleft()
            component.append((row, column))
            for neighbor_row, neighbor_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and painted[neighbor_row, neighbor_column]
                    and not visited[neighbor_row, neighbor_column]
                ):
                    visited[neighbor_row, neighbor_column] = True
                    pending.append((neighbor_row, neighbor_column))
        rows, columns = np.asarray(component, dtype=np.int64).T
        counts = np.bincount(region_labels[rows, columns])
        output[rows, columns] = int(np.argmax(counts))
    return output


def recolor_regions_preserve_luminance(
    original: np.ndarray,
    labels: np.ndarray,
    palette: dict[int, np.ndarray],
    strength: float = 1.0,
) -> np.ndarray:
    """Recenter each labeled region while retaining its local texture contrast."""
    image = np.asarray(original, dtype=np.float64)
    region_labels = np.asarray(labels, dtype=np.int64)
    if image.ndim != 3 or image.shape[2] == 0:
        raise ValueError("original must have shape (H, W, C)")
    if region_labels.shape != image.shape[:2]:
        raise ValueError("labels must match the image height and width")
    if not np.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between zero and one")
    output = image.copy()
    for region in np.unique(region_labels[region_labels >= 0]):
        key = int(region)
        if key not in palette:
            raise ValueError(f"missing palette color for region {key}")
        mask = region_labels == key
        target = np.asarray(palette[key], dtype=np.float64)
        if target.shape != (image.shape[2],) or not np.isfinite(target).all():
            raise ValueError(f"palette color for region {key} has the wrong shape")
        median = np.median(image[mask], axis=0)
        scale = target/np.maximum(median, 1.0e-6)
        recolored = image[mask]*scale
        output[mask] = (
            (1.0 - float(strength))*image[mask]
            + float(strength)*recolored
        )
    return np.clip(output, 0.0, 1.0)
