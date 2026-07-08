import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_dataset_root_defaults():
    from external_data_paths import dataset_root

    assert dataset_root("replicacad") == Path("/data/datasets/replica_cad")
    assert dataset_root("mixamo") == Path("/data/datasets/mixamo")


def test_dataset_root_env_override(monkeypatch, tmp_path):
    from external_data_paths import dataset_root

    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(tmp_path / "rc"))
    monkeypatch.setenv("AVENGINE_MIXAMO_ROOT", str(tmp_path / "mx"))

    assert dataset_root("replicacad") == tmp_path / "rc"
    assert dataset_root("mixamo") == tmp_path / "mx"


def test_require_dataset_root_missing(monkeypatch, tmp_path):
    from external_data_paths import DatasetMissingError, require_dataset_root

    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(tmp_path / "missing"))

    with pytest.raises(DatasetMissingError) as exc:
        require_dataset_root("replicacad")

    msg = str(exc.value)
    assert "AVENGINE_REPLICACAD_ROOT" in msg
    assert "ReplicaCAD" in msg


def test_unknown_dataset_name():
    from external_data_paths import dataset_root

    with pytest.raises(KeyError):
        dataset_root("unknown")
