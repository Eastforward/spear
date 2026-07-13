from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import run_controlled_animal_pixal_jobs as runner


def test_partition_jobs_balances_without_duplicates():
    jobs = [{"legacy_tag": f"animal_{index}"} for index in range(10)]

    partitions = runner.partition_jobs(jobs, [0, 1, 2, 3])

    assert [len(partitions[gpu]) for gpu in [0, 1, 2, 3]] == [3, 3, 2, 2]
    flattened = [job["legacy_tag"] for bucket in partitions.values() for job in bucket]
    assert sorted(flattened) == sorted(job["legacy_tag"] for job in jobs)


@pytest.mark.parametrize("gpus", [[], [0, 0], [0, 1, 2, 3, 4]])
def test_partition_jobs_rejects_invalid_gpu_contract(gpus):
    with pytest.raises(contracts.ContractError):
        runner.partition_jobs([{"legacy_tag": "animal"}], gpus)


def test_build_worker_job_separates_staging_write_from_public_path(tmp_path):
    public_root = tmp_path / "published"
    staging = tmp_path / ".published.staging"
    job = {
        "legacy_tag": "dog_x",
        "candidate_tag": "dog_x_pixal_v1",
        "seed": 42,
        "reference": {"pixal_input": {"path": "/input.png"}},
        "controlled_request": {
            "instance_id": "dog_x",
            "execution_job_id": "animal_x",
            "request_sha256": "0" * 64,
            "profile_schema_id": "dog_x_v1",
            "sampled_attributes": {"size": "small"},
            "target_physical_profile": {},
        },
    }

    worker_job = runner.build_worker_job(job, staging, public_root)

    assert Path(worker_job["output"]) == staging / "dog_x/pixal_raw_1024.glb"
    assert Path(worker_job["manifest"]) == staging / "dog_x/pixal_raw_1024.manifest.json"
    assert Path(worker_job["public_output"]) == public_root / "dog_x/pixal_raw_1024.glb"
    assert Path(worker_job["public_manifest"]) == public_root / "dog_x/pixal_raw_1024.manifest.json"
    assert worker_job["controlled_request"] == job["controlled_request"]
