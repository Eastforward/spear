import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_scene_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "replicaCAD.scene_dataset_config.json").write_text(json.dumps({
        "navmesh_instances": {
            "apt_0": "navmeshes/apt_0.navmesh",
            "empty_stage": "navmeshes/empty_stage.navmesh",
        },
        "scene_instances": {
            "paths": {".json": ["configs/scenes"]},
        },
    }))


def _write_scene_instance(root: Path, scene_id: str = "apt_0") -> None:
    scene = root / "configs" / "scenes" / f"{scene_id}.scene_instance.json"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(json.dumps({
        "stage_instance": {"template_name": "stages/frl_apartment_stage"},
        "default_lighting": "lighting/frl_apartment_stage",
        "object_instances": [
            {"template_name": "objects/chair", "motion_type": "DYNAMIC"},
            {"template_name": "objects/table", "motion_type": "DYNAMIC"},
        ],
        "articulated_object_instances": [
            {"template_name": "fridge"},
        ],
    }))


def _write_room_files(root: Path) -> None:
    _write_scene_dataset(root)
    _write_scene_instance(root)
    (root / "navmeshes").mkdir(parents=True, exist_ok=True)
    (root / "navmeshes" / "apt_0.navmesh").write_bytes(b"navmesh")
    (root / "stages").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "frl_apartment_stage.glb").write_bytes(b"glb")


def test_scene_id_maps_to_scene_instance_and_navmesh_paths(tmp_path):
    from replicacad_room_adapter import build_replicacad_room_import_plan

    root = tmp_path / "replica_cad"
    _write_room_files(root)

    plan = build_replicacad_room_import_plan(root, scene_id="apt_0")

    assert plan["state"] == "ready"
    assert plan["room_family"] == "replicacad"
    assert plan["scene_id"] == "apt_0"
    assert plan["scene_dataset_config_file"] == str(root / "replicaCAD.scene_dataset_config.json")
    assert plan["scene_instance_path"] == str(root / "configs" / "scenes" / "apt_0.scene_instance.json")
    assert plan["navmesh_path"] == str(root / "navmeshes" / "apt_0.navmesh")
    assert plan["stage_template"] == "stages/frl_apartment_stage"
    assert plan["stage_glb_path"] == str(root / "stages" / "frl_apartment_stage.glb")
    assert plan["habitat"]["scene_id"] == "apt_0"
    assert plan["habitat"]["scene_dataset_config_file"] == str(root / "replicaCAD.scene_dataset_config.json")
    assert plan["object_instance_count"] == 2
    assert plan["articulated_object_instance_count"] == 1


def test_room_import_plan_errors_when_navmesh_mapping_is_missing(tmp_path):
    from replicacad_room_adapter import ReplicaCADRoomImportError, build_replicacad_room_import_plan

    root = tmp_path / "replica_cad"
    _write_scene_dataset(root)
    _write_scene_instance(root, scene_id="apt_9")

    with pytest.raises(ReplicaCADRoomImportError, match="apt_9.*navmesh"):
        build_replicacad_room_import_plan(root, scene_id="apt_9")


def test_room_import_plan_errors_when_stage_glb_is_missing(tmp_path):
    from replicacad_room_adapter import ReplicaCADRoomImportError, build_replicacad_room_import_plan

    root = tmp_path / "replica_cad"
    _write_scene_dataset(root)
    _write_scene_instance(root)
    (root / "navmeshes").mkdir(parents=True, exist_ok=True)
    (root / "navmeshes" / "apt_0.navmesh").write_bytes(b"navmesh")

    with pytest.raises(ReplicaCADRoomImportError, match="stage.*frl_apartment_stage"):
        build_replicacad_room_import_plan(root, scene_id="apt_0")


def test_real_replicacad_apt0_layout_when_dataset_is_present():
    from external_data_paths import dataset_root
    from replicacad_room_adapter import build_replicacad_room_import_plan

    root = dataset_root("replicacad")
    if not root.exists():
        pytest.skip(f"ReplicaCAD root missing: {root}")

    plan = build_replicacad_room_import_plan(root, scene_id="apt_0")

    assert plan["state"] == "ready"
    assert Path(plan["scene_dataset_config_file"]).name == "replicaCAD.scene_dataset_config.json"
    assert Path(plan["scene_instance_path"]).name == "apt_0.scene_instance.json"
    assert Path(plan["navmesh_path"]).name == "apt_0.navmesh"
    assert Path(plan["stage_glb_path"]).exists()


def test_real_replicacad_habitat_apt0_loads_navmesh_when_available():
    habitat_sim = pytest.importorskip("habitat_sim")
    from external_data_paths import dataset_root
    from replicacad_room_adapter import build_replicacad_room_import_plan

    root = dataset_root("replicacad")
    if not root.exists():
        pytest.skip(f"ReplicaCAD root missing: {root}")
    plan = build_replicacad_room_import_plan(root, scene_id="apt_0")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = plan["habitat"]["scene_dataset_config_file"]
    sim_cfg.scene_id = plan["habitat"]["scene_id"]
    cfg = habitat_sim.Configuration(
        sim_cfg,
        [habitat_sim.agent.AgentConfiguration()],
    )
    sim = habitat_sim.Simulator(cfg)
    try:
        assert sim.pathfinder.load_nav_mesh(plan["habitat"]["explicit_navmesh_path"])
        point = sim.pathfinder.get_random_navigable_point()
        assert point.shape == (3,)
        assert np.all(np.isfinite(point))
    finally:
        sim.close()
