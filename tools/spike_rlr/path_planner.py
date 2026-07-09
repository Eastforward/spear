"""2D A* + Chaikin smoothing path planner.

Given start_xy, end_xy, and a list of blocked AABB rectangles (obstacles),
return a smooth N-point path from start to end that avoids all obstacles
plus optional wall-clearance buffer.

Algorithm (Plan 1, hand-tuned; Plan 2 will parameterize the grid resolution):
  1. Bake obstacles into an XY grid (default 20 cm cell).
  2. A* with 8-connected neighbors + Euclidean heuristic.
  3. Chaikin corner-cutting subdivision (2 iterations = smooth enough).
  4. Resample by uniform arc-length to exactly n_frames points.

The planner is intentionally 2D — Z is fixed (source_height_m). The Z
value is preserved as-is when returning the trajectory.
"""
from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np


def _point_in_any_rect(x, y, rects):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rects)


def _all_points_in_rects(points, rects):
    return all(_point_in_any_rect(float(x), float(y), rects) for x, y in points[:, :2])


def _bake_grid(x_min, y_min, x_max, y_max, cell_m, obstacles_xy, inflate_m=0.0,
               valid_xy_rects=None):
    """Return (grid, x_min, y_min, cell_m) where grid[i, j] = True means blocked.

    obstacles_xy: iterable of (x0, y0, x1, y1) rectangles in SSOT meters.
    inflate_m: additional buffer around each obstacle (Minkowski expansion).
    """
    nx = int(np.ceil((x_max - x_min) / cell_m))
    ny = int(np.ceil((y_max - y_min) / cell_m))
    grid = np.zeros((nx, ny), dtype=bool)
    if valid_xy_rects is not None:
        valid = [tuple(map(float, r)) for r in valid_xy_rects]
        for i in range(nx):
            for j in range(ny):
                x, y = _cell_to_xy(i, j, x_min, y_min, cell_m)
                if not _point_in_any_rect(x, y, valid):
                    grid[i, j] = True
    for x0, y0, x1, y1 in obstacles_xy:
        lo_i = max(0, int(np.floor((x0 - inflate_m - x_min) / cell_m)))
        hi_i = min(nx, int(np.ceil((x1 + inflate_m - x_min) / cell_m)))
        lo_j = max(0, int(np.floor((y0 - inflate_m - y_min) / cell_m)))
        hi_j = min(ny, int(np.ceil((y1 + inflate_m - y_min) / cell_m)))
        grid[lo_i:hi_i, lo_j:hi_j] = True
    return grid, x_min, y_min, cell_m


def _xy_to_cell(x, y, x_min, y_min, cell_m):
    return int(round((x - x_min) / cell_m)), int(round((y - y_min) / cell_m))


def _cell_to_xy(i, j, x_min, y_min, cell_m):
    return x_min + i * cell_m, y_min + j * cell_m


def _astar(grid, start_ij, goal_ij):
    """8-connected A*; returns list of (i, j) cells or None if unreachable."""
    nx, ny = grid.shape
    if not (0 <= start_ij[0] < nx and 0 <= start_ij[1] < ny):
        return None
    if not (0 <= goal_ij[0] < nx and 0 <= goal_ij[1] < ny):
        return None
    if grid[start_ij] or grid[goal_ij]:
        return None  # start or goal blocked

    def h(a, b):
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_heap = [(h(start_ij, goal_ij), 0.0, start_ij, None)]
    came_from = {}
    g_score = {start_ij: 0.0}

    while open_heap:
        _, g, cur, parent = heapq.heappop(open_heap)
        if cur in came_from:
            continue
        came_from[cur] = parent
        if cur == goal_ij:
            # Reconstruct
            path = [cur]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return list(reversed(path))
        for dx, dy in NEIGHBORS:
            nx_, ny_ = cur[0] + dx, cur[1] + dy
            if not (0 <= nx_ < nx and 0 <= ny_ < ny):
                continue
            if grid[nx_, ny_]:
                continue
            if dx != 0 and dy != 0:
                # No diagonal corner cutting through blocked/invalid cells.
                if grid[cur[0] + dx, cur[1]] or grid[cur[0], cur[1] + dy]:
                    continue
            step = (dx * dx + dy * dy) ** 0.5
            ng = g + step
            neighbor = (nx_, ny_)
            if neighbor in came_from or ng >= g_score.get(neighbor, float("inf")):
                continue
            g_score[neighbor] = ng
            heapq.heappush(open_heap, (ng + h(neighbor, goal_ij), ng, neighbor, cur))
    return None


def _chaikin(points, iterations=2):
    """Chaikin corner-cutting subdivision. Keeps endpoints; smooths interior."""
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for k in range(len(pts) - 1):
            p = pts[k]; q = pts[k + 1]
            new_pts.append(0.75 * p + 0.25 * q)   # 1/4 into segment
            new_pts.append(0.25 * p + 0.75 * q)   # 3/4 into segment
        new_pts.append(pts[-1])
        pts = np.asarray(new_pts)
    return pts


def _resample_arclength(points, n_frames):
    """Resample a polyline to exactly n_frames points uniformly by arc length."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        # Degenerate: same point n_frames times
        return np.tile(pts[:1], (n_frames, 1))
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(pts[:1], (n_frames, 1))
    targets = np.linspace(0.0, total, n_frames)
    out = np.empty((n_frames, pts.shape[1]))
    for d in range(pts.shape[1]):
        out[:, d] = np.interp(targets, cum, pts[:, d])
    return out


def plan_path_2d(start_xy, end_xy, obstacles_xy: Iterable[tuple],
                  bounds_xy: tuple, cell_m: float = 0.20,
                  inflate_m: float = 0.25, n_frames: int = 75,
                  chaikin_iters: int = 2, z_m: float | None = None,
                  valid_xy_rects: Iterable[tuple] | None = None):
    """Plan a smooth 2D path avoiding obstacles.

    Args:
      start_xy, end_xy: (x, y) in SSOT meters
      obstacles_xy: iterable of (x0, y0, x1, y1) blocked rectangles
      bounds_xy: (x_min, y_min, x_max, y_max) planning-region bounds
      cell_m: grid resolution (default 20 cm)
      inflate_m: how much to expand each obstacle (safety margin around
        family's body; default 25 cm ~ half of a small dog width)
      n_frames: number of output samples
      chaikin_iters: smoothing iterations (2 = very smooth, 0 = raw A*)
      z_m: Z coordinate to attach to every output point (None keeps 2D)
      valid_xy_rects: optional union of valid walkable rectangles. Any grid
        cell outside this union is blocked. This is for non-rectangular rooms
        like apartment_v1, whose outer bounds include outdoor voids.

    Returns:
      np.ndarray of shape (n_frames, 3) if z_m is not None else (n_frames, 2).
      Raises RuntimeError if no path found (caller may re-sample start/end).
    """
    x_min, y_min, x_max, y_max = bounds_xy
    if valid_xy_rects is not None:
        valid_xy_rects = [tuple(map(float, r)) for r in valid_xy_rects]
        if not _point_in_any_rect(start_xy[0], start_xy[1], valid_xy_rects):
            raise RuntimeError(f"start {start_xy} is outside valid region")
        if not _point_in_any_rect(end_xy[0], end_xy[1], valid_xy_rects):
            raise RuntimeError(f"end {end_xy} is outside valid region")
    grid, gx0, gy0, gc = _bake_grid(x_min, y_min, x_max, y_max, cell_m,
                                     obstacles_xy, inflate_m=inflate_m,
                                     valid_xy_rects=valid_xy_rects)
    s_ij = _xy_to_cell(start_xy[0], start_xy[1], gx0, gy0, gc)
    g_ij = _xy_to_cell(end_xy[0], end_xy[1], gx0, gy0, gc)
    if grid[s_ij[0], s_ij[1]]:
        raise RuntimeError(f"start {start_xy} is inside an inflated obstacle")
    if grid[g_ij[0], g_ij[1]]:
        raise RuntimeError(f"end {end_xy} is inside an inflated obstacle")

    cells = _astar(grid, s_ij, g_ij)
    if cells is None:
        raise RuntimeError(f"no path from {start_xy} to {end_xy} "
                            f"(grid {grid.shape}, obstacles={sum(1 for _ in obstacles_xy)})")

    # Cells -> world XY, snap start/end exactly.
    xy = np.array([_cell_to_xy(i, j, gx0, gy0, gc) for i, j in cells])
    xy[0] = start_xy
    xy[-1] = end_xy

    resampled = None
    for iters in range(chaikin_iters, -1, -1):
        candidate = _resample_arclength(_chaikin(xy, iterations=iters), n_frames)
        if valid_xy_rects is None or _all_points_in_rects(candidate, valid_xy_rects):
            resampled = candidate
            break
    if resampled is None:
        raise RuntimeError(
            f"no valid smoothed path from {start_xy} to {end_xy} inside valid region"
        )
    if z_m is not None:
        z_col = np.full((n_frames, 1), z_m, dtype=np.float64)
        return np.concatenate([resampled, z_col], axis=1)
    return resampled
