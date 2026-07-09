"""Downstream gate for approved mesh directory.

Import + call this from any pipeline that reads a Hunyuan mesh (blender_swap,
species_rig_map, run_render_pass_*.py). Raises with actionable message if
the mesh has not been human-approved via review_ui_server.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

try:
    from runtime_proxy_mesh import load_current_runtime_proxy_record
except ModuleNotFoundError:
    from .runtime_proxy_mesh import load_current_runtime_proxy_record


CURRENT_ALGORITHM_VERSION = "auto_orient_v1"


class MeshNotApprovedError(RuntimeError):
    """Raised when a downstream pipeline reads an unapproved Hunyuan mesh."""


def _default_approved_dir():
    return Path(__file__).resolve().parents[2] / "tmp" / "hy3d_batch" / "approved"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_mesh_approved(tag: str,
                          approved_dir: Optional[Path] = None,
                          required_algorithm_version: Optional[str] = None) -> dict:
    """Verify {approved_dir}/{tag}/direction.json exists + human_approved=True
    + algorithm_version matches + not quarantined. Returns loaded direction dict.

    Raises MeshNotApprovedError with an actionable message on any failure.
    """
    approved_dir = Path(approved_dir) if approved_dir else _default_approved_dir()
    required_algorithm_version = required_algorithm_version or CURRENT_ALGORITHM_VERSION

    tag_dir = approved_dir / tag
    dj_path = tag_dir / "direction.json"
    if not dj_path.exists():
        raise MeshNotApprovedError(
            f"Tag {tag!r} not found in approved/ ({dj_path}).\n"
            f"To fix: run the auto_orient_ingest pipeline on the source mesh, "
            f"then start review_ui_server.py and approve it in the browser."
        )
    d = json.loads(dj_path.read_text())

    if not d.get("human_approved"):
        raise MeshNotApprovedError(
            f"Tag {tag!r}: human_approved=False (mesh direction not yet "
            f"confirmed by a human).\n"
            f"To fix: start tools/spike_rlr/review_ui_server.py, open the web "
            f"UI (default http://localhost:8080/), and click Approve."
        )

    if d.get("quarantined"):
        raise MeshNotApprovedError(
            f"Tag {tag!r} is quarantined. Reason: "
            f"{d.get('quarantine_reason', 'unspecified')}.\n"
            f"To fix: manually edit {dj_path} to remove 'quarantined': true "
            f"after resolving the underlying issue, then re-review."
        )

    algo_v = d.get("algorithm_version")
    if algo_v != required_algorithm_version:
        raise MeshNotApprovedError(
            f"Tag {tag!r} was approved for algorithm_version={algo_v!r} but "
            f"pipeline requires {required_algorithm_version!r}.\n"
            f"To fix: re-run auto_orient_ingest --force on this tag, then "
            f"re-approve via review UI (algorithm has changed)."
        )

    return d


def resolve_approved_mesh_path(tag: str,
                                approved_dir: Optional[Path] = None) -> Path:
    """Return path to the CANONICAL (oriented) mesh for an approved tag."""
    approved_dir = Path(approved_dir) if approved_dir else _default_approved_dir()
    assert_mesh_approved(tag, approved_dir=approved_dir)
    # Prefer mesh_oriented.glb (already rotated to +X=head); fall back to mesh.glb
    for name in ("mesh_oriented.glb", "mesh.glb", "mesh.obj"):
        p = approved_dir / tag / name
        if p.exists():
            return p
    raise MeshNotApprovedError(
        f"Tag {tag!r} is approved but no mesh file found under {approved_dir / tag}"
    )


def approved_mesh_record(tag: str,
                         approved_dir: Optional[Path] = None) -> dict:
    """Return the approved canonical mesh plus provenance hash for one tag."""
    approved_dir = Path(approved_dir) if approved_dir else _default_approved_dir()
    direction = assert_mesh_approved(tag, approved_dir=approved_dir)
    mesh_path = resolve_approved_mesh_path(tag, approved_dir=approved_dir)
    actual_sha = sha256_file(mesh_path)
    recorded_sha = direction.get("mesh_sha256")
    if recorded_sha and recorded_sha != actual_sha:
        raise MeshNotApprovedError(
            f"Tag {tag!r}: mesh sha256 mismatch for {mesh_path}. "
            f"direction.json has {recorded_sha}, file has {actual_sha}."
        )
    runtime_rec = load_current_runtime_proxy_record(
        mesh_path.parent,
        source_mesh_sha256=actual_sha,
    )
    runtime_mesh_path = None
    runtime_mesh_sha256 = None
    if runtime_rec is not None:
        runtime_mesh_path = runtime_rec["runtime_mesh_path"]
        runtime_mesh_sha256 = runtime_rec["runtime_mesh_sha256"]
    return {
        "tag": tag,
        "mesh_path": mesh_path,
        "mesh_sha256": actual_sha,
        "runtime_mesh_path": runtime_mesh_path,
        "runtime_mesh_sha256": runtime_mesh_sha256,
        "direction_json_path": approved_dir / tag / "direction.json",
        "direction": direction,
    }
