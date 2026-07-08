import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_discover_replicacad_assets_counts_scene_and_mesh_files(tmp_path):
    from replicacad_probe import discover_replicacad_assets

    (tmp_path / "configs" / "scenes").mkdir(parents=True)
    (tmp_path / "meshes").mkdir()
    (tmp_path / "configs" / "scenes" / "apt.scene_instance.json").write_text("{}")
    (tmp_path / "meshes" / "apt.glb").write_text("glb")
    (tmp_path / "meshes" / "apt.ply").write_text("ply")
    (tmp_path / "README.md").write_text("ignore")

    inventory = discover_replicacad_assets(tmp_path)

    assert inventory.scene_instance_count == 1
    assert inventory.mesh_count == 2
    assert inventory.scene_instances == [
        tmp_path / "configs" / "scenes" / "apt.scene_instance.json"
    ]


def test_replicacad_probe_writes_missing_status(monkeypatch, tmp_path):
    from replicacad_probe import write_replicacad_probe_status

    missing = tmp_path / "missing"
    out = tmp_path / "status.json"
    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(missing))

    status = write_replicacad_probe_status(out)

    assert status["state"] == "missing_data"
    assert status["dataset"] == "replicacad"
    assert status["root"] == str(missing)
    assert "ReplicaCAD" in status["manual_action"]
    assert json.loads(out.read_text()) == status


def test_replicacad_probe_writes_ready_status(monkeypatch, tmp_path):
    from replicacad_probe import write_replicacad_probe_status

    root = tmp_path / "replicacad"
    (root / "configs" / "scenes").mkdir(parents=True)
    (root / "configs" / "scenes" / "room.scene_instance.json").write_text("{}")
    (root / "meshes").mkdir()
    (root / "meshes" / "room.glb").write_text("glb")
    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(root))

    status = write_replicacad_probe_status(tmp_path / "status.json")

    assert status["state"] == "ready"
    assert status["scene_instance_count"] == 1
    assert status["mesh_count"] == 1
    assert status["scene_instances"] == ["configs/scenes/room.scene_instance.json"]


def test_summarize_scene_instance_reports_stage_navmesh_and_counts(tmp_path):
    from replicacad_probe import summarize_scene_instance

    scene = tmp_path / "configs" / "scenes" / "room.scene_instance.json"
    scene.parent.mkdir(parents=True)
    scene.write_text(json.dumps({
        "stage_instance": {"template_name": "stages/frl_apartment_stage"},
        "navmesh_instance": "navmeshes/room.navmesh",
        "object_instances": [{"template_name": "objects/chair"}],
        "articulated_object_instances": [
            {"template_name": "fridge"},
            {"template_name": "cabinet"},
        ],
        "default_lighting": "lighting/frl_apartment_stage",
    }))

    summary = summarize_scene_instance(tmp_path, scene)

    assert summary == {
        "scene_instance": "configs/scenes/room.scene_instance.json",
        "stage_template": "stages/frl_apartment_stage",
        "navmesh_instance": "navmeshes/room.navmesh",
        "object_instance_count": 1,
        "articulated_object_instance_count": 2,
        "default_lighting": "lighting/frl_apartment_stage",
    }
