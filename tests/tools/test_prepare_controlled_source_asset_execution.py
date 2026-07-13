from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import controlled_animal_flux2_worker as animal_worker
from tools import execute_controlled_rocketbox_material_jobs as material_executor
from tools import prepare_controlled_source_asset_execution as execution
from tools import run_controlled_animal_flux2_jobs as animal_flux


REPO = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    REPO
    / "tmp/controlled_source_asset_input_v1/all_profiles_20260713_v3"
)
SCRIPT = REPO / "tools/prepare_controlled_source_asset_execution.py"


def test_preflight_reauthenticates_bundle_and_deduplicates_deterministic_materials():
    preflight = execution.build_execution_preflight(
        INPUT_ROOT,
        execution.default_artifact_roots(),
    )

    assert preflight["schema"] == "avengine_controlled_execution_preflight_v1"
    assert preflight["source_bundle"]["profile_count"] == 6
    assert preflight["source_bundle"]["request_count"] == 54
    assert preflight["source_bundle"]["planned_job_count"] == 54
    assert preflight["execution_summary"] == {
        "animal_job_count": 45,
        "deterministic_material_job_count": 3,
        "material_request_count": 9,
        "material_requests_deduplicated": 6,
        "unique_execution_job_count": 48,
    }
    animal_jobs = preflight["routes"]["flux2_pixal3d_animal_v1"]
    material_jobs = preflight["routes"]["rocketbox_material_v1"]
    assert len(animal_jobs) == 45
    assert len(material_jobs) == 3
    assert all(len(job["consumer_requests"]) == 1 for job in animal_jobs)
    assert all(len(job["consumer_requests"]) == 3 for job in material_jobs)
    assert {job["sampled_attributes"]["top_color"] for job in material_jobs} == {
        "blue",
        "green",
        "burgundy",
    }
    assert len({job["variant_key"] for job in material_jobs}) == 3
    assert len({job["variant_id"] for job in material_jobs}) == 3
    assert preflight["automatic_checks"] == {
        "all_profile_artifacts_authenticated": True,
        "all_requests_profile_validated": True,
        "execution_jobs_exactly_rebuilt": True,
        "material_jobs_deduplicated_by_absolute_plan": True,
        "overall": "passed",
    }
    assert preflight["preflight_sha256"] == execution.preflight_sha256(preflight)


def test_preflight_rejects_a_rehashed_but_noncanonical_execution_job(tmp_path):
    copied = tmp_path / "bundle"
    copied.mkdir()
    for name in execution.REQUIRED_INPUT_FILES:
        (copied / name).write_bytes((INPUT_ROOT / name).read_bytes())
    path = copied / "execution_jobs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"]["flux2_pixal3d_animal_v1"][0]["generation_plan"][
        "prompt"
    ] += " unauthorized adjective"
    payload["jobs_sha256"] = contracts.manifest_sha256(
        {key: value for key, value in payload.items() if key != "jobs_sha256"}
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        contracts.ContractError,
        match="execution_jobs.json does not exactly match",
    ):
        execution.build_execution_preflight(
            copied,
            execution.default_artifact_roots(),
        )


def test_preflight_cli_publishes_once_without_replacement(tmp_path):
    output = tmp_path / "preflight"
    command = [
        sys.executable,
        str(SCRIPT),
        "--input-dir",
        str(INPUT_ROOT),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CONTROLLED_EXECUTION_PREFLIGHT_OK" in completed.stdout
    manifest = json.loads((output / "execution_preflight.json").read_text())
    assert manifest["execution_summary"]["unique_execution_job_count"] == 48

    repeated = subprocess.run(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "refusing to replace" in repeated.stderr


def test_material_variant_request_is_runtime_builder_compatible():
    preflight = execution.build_execution_preflight(
        INPUT_ROOT,
        execution.default_artifact_roots(),
    )
    job = preflight["routes"]["rocketbox_material_v1"][0]
    texture = {
        "path": "variant/m002_body_color.tga",
        "sha256": "a" * 64,
        "size_bytes": 12_582_956,
    }

    request = execution.build_rocketbox_runtime_variant_request(job, texture)

    assert request["schema_version"] == "rocketbox_native_body_color_variant_v1"
    assert request["asset_id"] == "rocketbox_male_adult_01"
    assert request["variant_id"] == job["variant_id"]
    assert request["tag"] == f"rocketbox_male_adult_01_{job['variant_id']}"
    assert request["target_image_name"] == "m002_body_color"
    assert request["body_color_texture_sha256"] == "a" * 64
    assert request["body_color_texture_size_bytes"] == 12_582_956
    assert request["controlled_source"]["variant_key"] == job["variant_key"]
    assert len(request["controlled_source"]["consumer_requests"]) == 3


def test_material_executor_accepts_only_the_audited_fixed_geometry_plan():
    preflight = execution.build_execution_preflight(
        INPUT_ROOT,
        execution.default_artifact_roots(),
    )
    jobs = preflight["routes"]["rocketbox_material_v1"]

    assert [
        material_executor.validate_material_job(job)["variant_key"] for job in jobs
    ] == [job["variant_key"] for job in jobs]

    tampered = json.loads(json.dumps(jobs[0]))
    tampered["material_edit_plan"]["geometry_changes_allowed"] = True
    with pytest.raises(contracts.ContractError, match="contract changed"):
        material_executor.validate_material_job(tampered)


def test_animal_qa_canary_selects_one_single_attribute_pair_per_profile():
    preflight = execution.build_execution_preflight(
        INPUT_ROOT,
        execution.default_artifact_roots(),
    )

    jobs, pairs = animal_flux.select_qa_canary_jobs(preflight)

    assert len(jobs) == 10
    assert len(pairs) == 5
    assert len({pair["profile_schema_id"] for pair in pairs}) == 5
    assert all(len(pair["different_attributes"]) == 1 for pair in pairs)
    selected_instances = {
        job["consumer_requests"][0]["instance_id"] for job in jobs
    }
    assert all(
        {pair["instance_a"], pair["instance_b"]}.issubset(selected_instances)
        for pair in pairs
    )


def test_animal_worker_partition_pins_model_parameters_and_one_invocation():
    preflight = execution.build_execution_preflight(
        INPUT_ROOT,
        execution.default_artifact_roots(),
    )
    jobs, _pairs = animal_flux.select_qa_canary_jobs(
        preflight, profile_ids={"dog_golden_retriever_v1"}
    )
    partition = {
        "schema": animal_worker.PARTITION_SCHEMA,
        "execution_preflight_sha256": preflight["preflight_sha256"],
        "model": animal_flux.MODEL,
        "parameters": animal_flux.PARAMETERS,
        "jobs": jobs,
    }
    partition["partition_sha256"] = animal_flux._json_sha256(partition)

    assert animal_worker.validate_partition(partition) == partition

    tampered = json.loads(json.dumps(partition))
    tampered["jobs"][0]["generation_plan"]["flux_invocations"] = 2
    tampered["partition_sha256"] = animal_flux._json_sha256(
        {key: value for key, value in tampered.items() if key != "partition_sha256"}
    )
    with pytest.raises(ValueError, match="animal job contract"):
        animal_worker.validate_partition(tampered)
