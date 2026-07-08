"""Probe locally available ReplicaCAD scene assets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from external_data_paths import dataset_root, dataset_spec


@dataclass(frozen=True)
class ReplicaCADInventory:
    scene_instances: list[Path]
    meshes: list[Path]

    @property
    def scene_instance_count(self) -> int:
        return len(self.scene_instances)

    @property
    def mesh_count(self) -> int:
        return len(self.meshes)


def _sorted_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in root.rglob(pattern) if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(root).as_posix().lower())


def discover_replicacad_assets(root: Path) -> ReplicaCADInventory:
    return ReplicaCADInventory(
        scene_instances=_sorted_files(root, ("*.scene_instance.json",)),
        meshes=_sorted_files(root, ("*.glb", "*.ply")),
    )


def summarize_scene_instance(root: Path, scene_instance: Path) -> dict:
    data = json.loads(Path(scene_instance).read_text())
    stage = data.get("stage_instance") or {}
    object_instances = data.get("object_instances") or []
    articulated = data.get("articulated_object_instances") or []
    return {
        "scene_instance": Path(scene_instance).relative_to(root).as_posix(),
        "stage_template": stage.get("template_name"),
        "navmesh_instance": data.get("navmesh_instance"),
        "object_instance_count": len(object_instances),
        "articulated_object_instance_count": len(articulated),
        "default_lighting": data.get("default_lighting"),
    }


def write_replicacad_probe_status(out_path: Path) -> dict:
    spec = dataset_spec("replicacad")
    root = dataset_root("replicacad")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        status = {
            "state": "missing_data",
            "dataset": spec.name,
            "root": str(root),
            "manual_action": spec.acquisition_hint,
        }
    else:
        inventory = discover_replicacad_assets(root)
        state = "ready" if (inventory.scene_instances or inventory.meshes) else "no_assets"
        status = {
            "state": state,
            "dataset": spec.name,
            "root": str(root),
            "scene_instance_count": inventory.scene_instance_count,
            "mesh_count": inventory.mesh_count,
            "scene_instances": [
                p.relative_to(root).as_posix() for p in inventory.scene_instances[:20]
            ],
            "meshes": [p.relative_to(root).as_posix() for p in inventory.meshes[:20]],
        }
        if inventory.scene_instances:
            status["first_scene_summary"] = summarize_scene_instance(
                root, inventory.scene_instances[0]
            )

    out_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp/replicacad_probe/status.json"))
    args = parser.parse_args(argv)
    status = write_replicacad_probe_status(args.out)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
