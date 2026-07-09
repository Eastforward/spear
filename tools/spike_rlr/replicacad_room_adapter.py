"""ReplicaCAD room import-plan helpers.

This is the lightweight contract layer between ReplicaCAD's Habitat dataset
layout and future UE/RLR import code. It validates the scene id, scene instance,
stage GLB, and navmesh paths without launching Habitat or Unreal.
"""
from __future__ import annotations

import json
from pathlib import Path


class ReplicaCADRoomImportError(RuntimeError):
    """Raised when a ReplicaCAD scene cannot form a complete import plan."""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ReplicaCADRoomImportError(f"missing ReplicaCAD file: {path}") from exc


def scene_instance_path(root: Path, scene_id: str) -> Path:
    return Path(root) / "configs" / "scenes" / f"{scene_id}.scene_instance.json"


def scene_dataset_config_path(root: Path) -> Path:
    return Path(root) / "replicaCAD.scene_dataset_config.json"


def _navmesh_path(root: Path, scene_id: str, dataset_config: dict) -> Path:
    navmesh_rel = (dataset_config.get("navmesh_instances") or {}).get(scene_id)
    if not navmesh_rel:
        raise ReplicaCADRoomImportError(
            f"ReplicaCAD scene {scene_id!r} has no navmesh mapping in "
            f"{scene_dataset_config_path(root)}"
        )
    path = Path(root) / navmesh_rel
    if not path.exists():
        raise ReplicaCADRoomImportError(
            f"ReplicaCAD scene {scene_id!r} navmesh does not exist: {path}"
        )
    return path


def _stage_glb_path(root: Path, stage_template: str | None) -> Path:
    if not stage_template:
        raise ReplicaCADRoomImportError("ReplicaCAD scene has no stage template")
    path = Path(root) / f"{stage_template}.glb"
    if not path.exists():
        raise ReplicaCADRoomImportError(
            f"ReplicaCAD stage GLB for {stage_template!r} does not exist: {path}"
        )
    return path


def build_replicacad_room_import_plan(root: Path, *, scene_id: str = "apt_0") -> dict:
    """Validate and describe the paths needed to import one ReplicaCAD room."""
    root = Path(root)
    if not root.exists():
        raise ReplicaCADRoomImportError(f"ReplicaCAD root does not exist: {root}")

    config_path = scene_dataset_config_path(root)
    scene_path = scene_instance_path(root, scene_id)
    dataset_config = _read_json(config_path)
    scene = _read_json(scene_path)
    navmesh = _navmesh_path(root, scene_id, dataset_config)

    stage_template = (scene.get("stage_instance") or {}).get("template_name")
    stage_glb = _stage_glb_path(root, stage_template)
    object_instances = scene.get("object_instances") or []
    articulated = scene.get("articulated_object_instances") or []

    return {
        "state": "ready",
        "room_family": "replicacad",
        "scene_id": scene_id,
        "root": str(root),
        "scene_dataset_config_file": str(config_path),
        "scene_instance_path": str(scene_path),
        "navmesh_path": str(navmesh),
        "stage_template": stage_template,
        "stage_glb_path": str(stage_glb),
        "default_lighting": scene.get("default_lighting"),
        "object_instance_count": len(object_instances),
        "articulated_object_instance_count": len(articulated),
        "habitat": {
            "scene_id": scene_id,
            "scene_dataset_config_file": str(config_path),
            "explicit_navmesh_path": str(navmesh),
        },
        "coordinate_convention": {
            "dataset_frame": "ReplicaCAD/Habitat native Y-up",
            "avengine_scene_frame": "pending_adapter_transform",
            "notes": (
                "This plan validates source files only; UE/RLR transform "
                "calibration must be proven by the later smoke clip."
            ),
        },
    }
