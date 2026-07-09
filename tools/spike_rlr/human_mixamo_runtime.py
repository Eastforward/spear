"""Prepare approved Flux/Hunyuan human meshes for Mixamo runtime use."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from external_data_paths import dataset_root
from mixamo_humanoid_adapter import build_mixamo_humanoid_import_plan


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_SCHEMA_VERSION = "human_mixamo_runtime_v1"
RUNTIME_TYPE = "mixamo_humanoid_nearest_skin_transfer"


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _clip_source_fbx(plan: dict[str, Any], role: str) -> str | None:
    clip = (plan.get("clips") or {}).get(role) or {}
    return clip.get("fbx_relative_path") or clip.get("fbx_path")


def prepare_human_mixamo_runtime(
    tag_dir: Path | str,
    *,
    walking_glb: Path | str,
    idle_glb: Path | str,
    mixamo_root: Path | str | None = None,
    recommended_actor_scale: float = 1.0,
    recommended_actor_z_lift_cm: float = 0.0,
    recommended_walking_forward_yaw_offset_deg: float = 90.0,
) -> Path:
    """Copy verified Mixamo-skinned GLBs into an approved human tag directory.

    ``mesh_runtime.glb`` remains a compatibility alias for the walking runtime.
    Per-animation GLBs are recorded in ``mesh_runtime.json`` so event builders
    can choose a loopable clip without guessing.
    """
    tag_dir = Path(tag_dir)
    walking_src = Path(walking_glb)
    idle_src = Path(idle_glb)
    mixamo_root = Path(mixamo_root) if mixamo_root is not None else dataset_root("mixamo")

    walking_dst = tag_dir / "mesh_runtime_walking.glb"
    idle_dst = tag_dir / "mesh_runtime_standing_idle.glb"
    legacy_dst = tag_dir / "mesh_runtime.glb"
    _copy(walking_src, walking_dst)
    _copy(idle_src, idle_dst)
    _copy(walking_dst, legacy_dst)

    plan = build_mixamo_humanoid_import_plan(mixamo_root)
    metadata = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_type": RUNTIME_TYPE,
        "default_animation": "Walking",
        "recommended_actor_scale": float(recommended_actor_scale),
        "recommended_actor_z_lift_cm": float(recommended_actor_z_lift_cm),
        "recommended_walking_forward_yaw_offset_deg": float(
            recommended_walking_forward_yaw_offset_deg
        ),
        "legacy_runtime": _repo_path(walking_dst),
        "animations": {
            "Standing_Idle": {
                "role": "idle",
                "motion_style": "stationary",
                "loop": True,
                "glb_path": _repo_path(idle_dst),
                "source_glb": _repo_path(idle_src),
                "source_fbx": _clip_source_fbx(plan, "idle"),
                "sha256": _sha256_file(idle_dst),
            },
            "Walking": {
                "role": "walk",
                "motion_style": "walking",
                "loop": True,
                "glb_path": _repo_path(walking_dst),
                "source_glb": _repo_path(walking_src),
                "source_fbx": _clip_source_fbx(plan, "walk"),
                "sha256": _sha256_file(walking_dst),
            },
        },
    }
    path = tag_dir / "mesh_runtime.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
