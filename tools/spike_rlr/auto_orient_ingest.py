"""Batch ingest driver for Hunyuan mesh directory.

For each tag directory under --pending-dir:
  1. Load mesh.glb (or mesh.obj)
  2. Run detect_head_axis
  3. Compute rotation matrix to align head with +X
  4. Write mesh_oriented.glb (rotated)
  5. Write direction_preview.png (Task 3)
  6. Write direction.json with human_approved=False

Idempotent by default: if direction.json exists, skip the tag. Use --force
to re-process (bumps algorithm_version if different).

Usage:
  /data/jzy/miniconda3/envs/ss2/bin/python \\
      tools/spike_rlr/auto_orient_ingest.py \\
      --pending-dir tmp/hy3d_batch/pending
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Keep per-mesh audit preparation cheap and predictable. Batch pipelines can
# parallelize across assets without every worker spawning its own BLAS pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))

from detect_head_axis import HeadDetectionResult, detect_head_axis  # noqa: E402
from preview_render import render_direction_preview  # noqa: E402


ALGORITHM_VERSION = "auto_orient_v1"


def _rel_or_abs(p: Path) -> str:
    """Return path relative to REPO_ROOT if possible, else absolute string."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _find_mesh_file(tag_dir: Path) -> Path:
    for name in ("mesh.glb", "mesh.obj"):
        p = tag_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no mesh.glb or mesh.obj in {tag_dir}")


def _candidate_category(tag_dir: Path) -> str | None:
    path = tag_dir / "source_asset_candidate.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    category = data.get("category")
    return str(category).lower() if category else None


def _load_and_concat(mesh_path: Path):
    scene = trimesh.load(str(mesh_path))
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        if not geoms:
            raise ValueError(f"empty scene {mesh_path}")
        return trimesh.util.concatenate(geoms)
    return scene


def _rotated_mesh_preserving_visuals(mesh: trimesh.Trimesh, R: np.ndarray):
    """Rotate vertices while keeping UV/material/visual data attached."""
    rotated = mesh.copy()
    rotated.vertices = np.asarray(mesh.vertices) @ R.T
    return rotated


def _rotation_matrix_align(from_vec, to_vec):
    """Compute the 3x3 rotation matrix that rotates from_vec to to_vec.

    Uses Rodrigues' rotation formula. Handles the antiparallel edge case
    (180-degree rotation) by picking any orthogonal axis.
    """
    f = np.asarray(from_vec, dtype=np.float64)
    t = np.asarray(to_vec, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    t = t / (np.linalg.norm(t) + 1e-12)
    v = np.cross(f, t)
    s = np.linalg.norm(v)
    c = float(np.dot(f, t))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)  # parallel already
        # antiparallel — rotate 180 about any axis orthogonal to f
        # Pick the smallest component axis to build an orthogonal
        smallest = np.argmin(np.abs(f))
        e = np.zeros(3); e[smallest] = 1.0
        axis = np.cross(f, e); axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _manual_human_orientation_result() -> HeadDetectionResult:
    return HeadDetectionResult(
        head_direction=np.array([1.0, 0.0, 0.0]),
        signals={"manual_human_orientation_required": 1},
        total_votes=0,
        unanimous=False,
        confidence=0.0,
        pc1_axis=np.array([1.0, 0.0, 0.0]),
        pc2_axis=np.array([0.0, 1.0, 0.0]),
    )


def _write_manual_human_orientation(tag_dir: Path, mesh_path: Path) -> str:
    result = _manual_human_orientation_result()
    stale_oriented = tag_dir / "mesh_oriented.glb"
    if stale_oriented.exists():
        stale_oriented.unlink()

    preview_path = tag_dir / "direction_preview.png"
    render_direction_preview(mesh_path, result, preview_path)

    payload = {
        "mesh_source": _rel_or_abs(mesh_path),
        "mesh_oriented": None,
        "algorithm_version": ALGORITHM_VERSION,
        "orientation_strategy": "manual_human_orientation_v1",
        "auto_orientation_skipped": True,
        "detected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "detection": {
            "head_direction_original_mesh_frame": result.head_direction.tolist(),
            "rotation_applied_to_align_to_plus_x": np.eye(3).tolist(),
            "signals": result.signals,
            "total_votes": result.total_votes,
            "unanimous": result.unanimous,
            "confidence": result.confidence,
        },
        "human_approved": False,
        "human_approved_by": None,
        "human_approved_at": None,
        "human_notes": "human asset: animal auto-orient skipped; use review UI manual rotation",
        "human_override": None,
        "quarantined": False,
    }
    (tag_dir / "direction.json").write_text(json.dumps(payload, indent=2))
    print(
        f"  {tag_dir.name}: human category -> skipped animal auto-orient; "
        f"wrote direction.json for manual review"
    )
    return "processed"


def process_one(tag_dir: Path, force: bool = False):
    dj_path = tag_dir / "direction.json"
    if dj_path.exists() and not force:
        print(f"  {tag_dir.name}: direction.json exists, skipping (use --force to redo)")
        return "skipped"

    mesh_path = _find_mesh_file(tag_dir)
    if _candidate_category(tag_dir) == "human":
        return _write_manual_human_orientation(tag_dir, mesh_path)

    mesh = _load_and_concat(mesh_path)
    verts = np.array(mesh.vertices)
    result = detect_head_axis(verts)

    # Rotate mesh so detected head aligns with +X
    R = _rotation_matrix_align(result.head_direction, np.array([1.0, 0.0, 0.0]))
    oriented = _rotated_mesh_preserving_visuals(mesh, R)
    oriented_path = tag_dir / "mesh_oriented.glb"
    oriented.export(str(oriented_path))

    # Write preview PNG (renders the ORIGINAL mesh with arrow, so reviewer
    # sees which end IS the head in the source mesh — not the oriented one)
    preview_path = tag_dir / "direction_preview.png"
    render_direction_preview(mesh_path, result, preview_path)

    payload = {
        "mesh_source": _rel_or_abs(mesh_path),
        "mesh_oriented": _rel_or_abs(oriented_path),
        "algorithm_version": ALGORITHM_VERSION,
        "detected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "detection": {
            "head_direction_original_mesh_frame": result.head_direction.tolist(),
            "rotation_applied_to_align_to_plus_x": R.tolist(),
            "signals": result.signals,
            "total_votes": result.total_votes,
            "unanimous": result.unanimous,
            "confidence": result.confidence,
        },
        "human_approved": False,
        "human_approved_by": None,
        "human_approved_at": None,
        "human_notes": None,
        "human_override": None,
        "quarantined": False,
    }
    dj_path.write_text(json.dumps(payload, indent=2))
    print(f"  {tag_dir.name}: head={result.head_direction} "
          f"conf={result.confidence:.0%} unanimous={result.unanimous} "
          f"-> wrote {dj_path.name}, mesh_oriented.glb, direction_preview.png")
    return "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending-dir", required=True,
                    help="Path to pending/ containing per-tag subdirectories")
    ap.add_argument("--force", action="store_true",
                    help="Re-process tags even if direction.json exists")
    args = ap.parse_args()

    pending = Path(args.pending_dir)
    if not pending.exists():
        raise SystemExit(f"pending dir does not exist: {pending}")

    tag_dirs = [d for d in pending.iterdir()
                if d.is_dir() and not d.name.startswith(".")]
    if not tag_dirs:
        print(f"No tag directories found under {pending}")
        return

    print(f"Found {len(tag_dirs)} pending tag(s):")
    processed = skipped = failed = 0
    for tag_dir in sorted(tag_dirs):
        try:
            status = process_one(tag_dir, force=args.force)
            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1
        except Exception as e:
            print(f"  {tag_dir.name}: FAILED -- {e}")
            failed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
