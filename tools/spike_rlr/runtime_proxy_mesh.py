"""Runtime proxy mesh sidecar helpers.

The human audit source remains mesh_oriented.glb.  mesh_runtime.glb is an
optional derived asset for expensive downstream Blender/UE import paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


RUNTIME_PROXY_ALGORITHM = "blender_decimate_v1"
RUNTIME_PROXY_MESH_NAME = "mesh_runtime.glb"
RUNTIME_PROXY_META_NAME = "mesh_runtime.json"
DEFAULT_TARGET_FACES = 40000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_proxy_paths(tag_dir: Path) -> tuple[Path, Path]:
    tag_dir = Path(tag_dir)
    return tag_dir / RUNTIME_PROXY_MESH_NAME, tag_dir / RUNTIME_PROXY_META_NAME


def load_current_runtime_proxy_record(
    tag_dir: Path,
    *,
    source_mesh_sha256: str,
    target_faces: Optional[int] = None,
) -> Optional[dict]:
    runtime_mesh, runtime_meta = runtime_proxy_paths(tag_dir)
    if not runtime_mesh.exists() or not runtime_meta.exists():
        return None

    try:
        rec = json.loads(runtime_meta.read_text())
    except Exception:
        return None

    if rec.get("algorithm") != RUNTIME_PROXY_ALGORITHM:
        return None
    if rec.get("source_mesh_sha256") != source_mesh_sha256:
        return None
    if target_faces is not None and int(rec.get("target_faces", -1)) != int(target_faces):
        return None

    runtime_sha = sha256_file(runtime_mesh)
    recorded_runtime_sha = rec.get("runtime_mesh_sha256")
    if recorded_runtime_sha and recorded_runtime_sha != runtime_sha:
        return None

    out = dict(rec)
    out["runtime_mesh_path"] = runtime_mesh
    out["runtime_mesh_sha256"] = runtime_sha
    return out


def write_runtime_proxy_record(
    *,
    metadata_path: Path,
    source_mesh_path: Path,
    runtime_mesh_path: Path,
    target_faces: int,
    source_faces: int,
    source_vertices: int,
    actual_faces: int,
    actual_vertices: int,
) -> dict:
    rec = {
        "algorithm": RUNTIME_PROXY_ALGORITHM,
        "source_mesh": str(Path(source_mesh_path).resolve()),
        "source_mesh_sha256": sha256_file(Path(source_mesh_path)),
        "runtime_mesh": str(Path(runtime_mesh_path).resolve()),
        "runtime_mesh_sha256": sha256_file(Path(runtime_mesh_path)),
        "target_faces": int(target_faces),
        "source_faces": int(source_faces),
        "source_vertices": int(source_vertices),
        "actual_faces": int(actual_faces),
        "actual_vertices": int(actual_vertices),
    }
    Path(metadata_path).write_text(json.dumps(rec, indent=2))
    return rec
