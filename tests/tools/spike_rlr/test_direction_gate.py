import json
import sys
from pathlib import Path

import pytest
import trimesh


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_box(path: Path, extents):
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(path)


def test_direction_gate_passes_elongated_approved_mesh(tmp_path):
    from direction_gate import direction_gate_report

    mesh_path = tmp_path / "animal.glb"
    direction_path = tmp_path / "direction.json"
    _write_box(mesh_path, (2.0, 0.6, 0.8))
    direction_path.write_text(json.dumps({"human_approved": True}))

    report = direction_gate_report(mesh_path, direction_path)

    assert report["decision"] == "pass"
    assert report["confidence"] >= 0.8
    assert report["checks"]["horizontal_major_minor_ratio"] > 1.2


def test_direction_gate_blocks_symmetric_mesh(tmp_path):
    from direction_gate import direction_gate_report

    mesh_path = tmp_path / "cube.glb"
    direction_path = tmp_path / "direction.json"
    _write_box(mesh_path, (1.0, 1.0, 1.0))
    direction_path.write_text(json.dumps({"human_approved": True}))

    report = direction_gate_report(mesh_path, direction_path)

    assert report["decision"] == "block"
    assert report["confidence"] < 0.8
    assert "near_symmetric_horizontal_bounds" in report["reasons"]


def test_direction_gate_requires_mesh(tmp_path):
    from direction_gate import direction_gate_report

    with pytest.raises(FileNotFoundError):
        direction_gate_report(tmp_path / "missing.glb")


def test_write_direction_gate_report_creates_json(tmp_path):
    from direction_gate import write_direction_gate_report

    mesh_path = tmp_path / "animal.glb"
    out_path = tmp_path / "report.json"
    _write_box(mesh_path, (2.0, 0.6, 0.8))

    report = write_direction_gate_report(mesh_path, out_path)

    assert json.loads(out_path.read_text()) == report
    assert report["mesh_path"] == str(mesh_path)
    assert "checks" in report
    assert "decision" in report
