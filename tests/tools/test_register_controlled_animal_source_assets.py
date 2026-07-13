from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as registry


def test_spear_artifact_rejects_paths_outside_repo(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"fixture")

    with pytest.raises(contracts.ContractError, match="outside SPEAR"):
        registry.spear_artifact(path)


def test_pinned_model_license_snapshots_are_present_and_hashed():
    records = registry.license_records()

    assert len(records) == 4
    assert {record["root_id"] for record in records} == {"models_root"}
    assert all(len(record["sha256"]) == 64 for record in records)
