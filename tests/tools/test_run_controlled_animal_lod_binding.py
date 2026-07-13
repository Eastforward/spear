import json
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import run_controlled_animal_lod_binding as runner


def _job(tmp_path, species="dog"):
    spec = runner.RIG_SPECS[species]
    return {
        "raw_path": tmp_path / "raw.glb",
        "rig": {**spec, "path": Path(spec["path"]), "species": species},
    }


def test_build_commands_pins_approved_lod_and_binding_contract(tmp_path):
    lod, bind = runner.build_commands(_job(tmp_path), tmp_path / "job", target_faces=100_000)

    assert lod[-3:] == ["--target-faces", "100000", "--double-sided"]
    assert "--double-sided" in lod
    assert bind[bind.index("--rig-glb") + 1].endswith("Dog.glb")
    assert "--flip-x" in bind
    assert bind[bind.index("--align-mode") + 1] == "uniform"
    assert bind[bind.index("--weight-mode") + 1] == "region"
    assert bind[bind.index("--segmentation-mode") + 1] == "proximity"
    assert bind[bind.index("--semantic-forward-axis") + 1] == "positive-x"
    assert bind[bind.index("--export-action-policy") + 1] == "walk-idle"


def test_rig_spec_rejects_profile_and_species_mismatch():
    source = {
        "asset_id": "animal_x",
        "taxonomy": {"species": "cat"},
        "rig": {
            "profile_id": "quadruped_dog_v1",
            "skeleton_family": "quaternius_cat",
            "front_axis": "positive_x",
            "actions": ["Walking", "Idle"],
        },
    }

    with pytest.raises(contracts.ContractError, match="rig contract"):
        runner._rig_spec(source)


def test_rewrite_runtime_metadata_records_published_path(tmp_path):
    runtime = tmp_path / "runtime.glb"
    runtime.write_bytes(b"glb fixture")
    source = tmp_path / "source.glb"
    source.write_bytes(b"source fixture")
    metadata = tmp_path / "runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "algorithm": "blender_decimate_v1",
                "source_mesh_sha256": runner._sha256_file(source),
                "runtime_mesh": str(runtime),
                "runtime_mesh_sha256": runner._sha256_file(runtime),
            }
        )
    )
    published = tmp_path / "published/runtime.glb"

    runner._rewrite_runtime_metadata(
        metadata, published, source_sha256=runner._sha256_file(source)
    )

    assert json.loads(metadata.read_text())["runtime_mesh"] == str(published.resolve())


@pytest.mark.parametrize("workers", [0, 17])
def test_run_batch_rejects_unsafe_worker_counts(tmp_path, workers):
    with pytest.raises(contracts.ContractError, match="workers"):
        runner.run_batch([], tmp_path / "output", workers=workers)


def test_load_jobs_requires_a_registry():
    with pytest.raises(contracts.ContractError, match="registry"):
        runner.load_jobs([])
