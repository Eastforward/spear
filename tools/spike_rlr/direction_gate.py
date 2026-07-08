"""Automatic, conservative direction-confidence gate for approved meshes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _load_bounds(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(mesh_path, force="scene")
    if hasattr(loaded, "bounds") and loaded.bounds is not None:
        bounds = np.asarray(loaded.bounds, dtype=float)
    elif hasattr(loaded, "geometry") and loaded.geometry:
        mins = []
        maxs = []
        for geom in loaded.geometry.values():
            gb = np.asarray(geom.bounds, dtype=float)
            mins.append(gb[0])
            maxs.append(gb[1])
        bounds = np.asarray([np.min(mins, axis=0), np.max(maxs, axis=0)])
    else:
        raise ValueError(f"Mesh has no usable bounds: {mesh_path}")
    extents = bounds[1] - bounds[0]
    return bounds, extents


def _load_direction_metadata(direction_json: Path | None) -> dict[str, Any]:
    if direction_json is None or not direction_json.exists():
        return {}
    return json.loads(direction_json.read_text())


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(float(value), 6)


def direction_gate_report(
    mesh_path: Path,
    direction_json: Path | None = None,
) -> dict[str, Any]:
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    _, extents = _load_bounds(mesh_path)
    x_extent, y_extent, z_extent = [float(v) for v in extents]
    horizontal = sorted([abs(x_extent), abs(y_extent)], reverse=True)
    ratio = horizontal[0] / max(horizontal[1], 1e-9)
    finite_extents = all(math.isfinite(float(v)) and float(v) > 0 for v in extents)

    direction_meta = _load_direction_metadata(direction_json)
    has_human_direction = bool(
        direction_meta.get("human_approved")
        or direction_meta.get("human_applied_rotation_matrix")
    )

    reasons: list[str] = []
    confidence = 0.0
    if finite_extents:
        confidence += 0.1
    else:
        reasons.append("invalid_bounds")

    if ratio >= 1.2:
        confidence += 0.55
    else:
        reasons.append("near_symmetric_horizontal_bounds")

    if has_human_direction:
        confidence += 0.35
    else:
        reasons.append("no_human_direction_metadata")

    confidence = min(confidence, 1.0)
    decision = "pass" if confidence >= 0.8 and not reasons else "block"

    return {
        "mesh_path": str(mesh_path),
        "direction_json": str(direction_json) if direction_json else None,
        "decision": decision,
        "confidence": _round_float(confidence),
        "reasons": reasons,
        "checks": {
            "x_extent_m": _round_float(x_extent),
            "y_extent_m": _round_float(y_extent),
            "z_extent_m": _round_float(z_extent),
            "horizontal_major_minor_ratio": _round_float(ratio),
            "finite_extents": finite_extents,
            "has_human_direction_metadata": has_human_direction,
        },
    }


def write_direction_gate_report(
    mesh_path: Path,
    out_path: Path,
    direction_json: Path | None = None,
) -> dict[str, Any]:
    report = direction_gate_report(mesh_path, direction_json)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
