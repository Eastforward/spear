import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_discover_mixamo_fbx_returns_sorted_files(tmp_path):
    from mixamo_probe import discover_mixamo_fbx

    (tmp_path / "Jump").mkdir()
    (tmp_path / "Walk").mkdir()
    (tmp_path / "Walk" / "Walk.fbx").write_text("fbx")
    (tmp_path / "Jump" / "Jump.FBX").write_text("fbx")
    (tmp_path / "Walk" / "notes.txt").write_text("ignore")

    files = discover_mixamo_fbx(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in files] == [
        "Jump/Jump.FBX",
        "Walk/Walk.fbx",
    ]


def test_mixamo_probe_writes_missing_status(monkeypatch, tmp_path):
    from mixamo_probe import write_mixamo_probe_status

    missing = tmp_path / "missing"
    out = tmp_path / "status.json"
    monkeypatch.setenv("AVENGINE_MIXAMO_ROOT", str(missing))

    status = write_mixamo_probe_status(out)

    assert status["state"] == "missing_data"
    assert status["dataset"] == "mixamo"
    assert status["root"] == str(missing)
    assert "Mixamo" in status["manual_action"]
    assert json.loads(out.read_text()) == status


def test_mixamo_probe_writes_ready_status(monkeypatch, tmp_path):
    from mixamo_probe import write_mixamo_probe_status

    root = tmp_path / "mixamo"
    (root / "Idle").mkdir(parents=True)
    (root / "Idle" / "Idle.fbx").write_text("fbx")
    monkeypatch.setenv("AVENGINE_MIXAMO_ROOT", str(root))

    status = write_mixamo_probe_status(tmp_path / "status.json")

    assert status["state"] == "ready"
    assert status["fbx_count"] == 1
    assert status["fbx_files"] == ["Idle/Idle.fbx"]
