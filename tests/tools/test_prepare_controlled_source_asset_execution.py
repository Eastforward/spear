from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import controlled_animal_flux2_worker as animal_worker
from tools import controlled_animal_one_shot_policy as one_shot
from tools import execute_controlled_rocketbox_material_jobs as material_executor
from tools import prepare_controlled_source_asset_execution as execution
from tools import run_controlled_animal_flux2_jobs as animal_flux


REPO = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    REPO
    / "tmp/controlled_source_asset_input_v1/all_profiles_20260713_v3"
)
SCRIPT = REPO / "tools/prepare_controlled_source_asset_execution.py"


def _artifact_record(root_id: str, path: str, payload: bytes) -> dict:
    return {
        "root_id": root_id,
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_artifact_authentication_allows_repository_tmp_compatibility_mount(
    tmp_path,
):
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repo.mkdir()
    workspace.mkdir()
    (repo / "tmp").symlink_to(workspace, target_is_directory=True)
    payload = b"authenticated workspace artifact\n"
    artifact = workspace / "evidence.bin"
    artifact.write_bytes(payload)

    result = input_builder.authenticate_artifact_record(
        _artifact_record("spear_repo", "tmp/evidence.bin", payload),
        {"spear_repo": repo},
        role="fixture",
        owner="compatibility mount test",
    )

    assert result["status"] == "passed"
    assert result["path"] == "tmp/evidence.bin"


def test_artifact_authentication_rejects_undeclared_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "other").symlink_to(outside, target_is_directory=True)
    payload = b"outside artifact\n"
    (outside / "evidence.bin").write_bytes(payload)

    with pytest.raises(contracts.ContractError, match="artifact escapes root"):
        input_builder.authenticate_artifact_record(
            _artifact_record("spear_repo", "other/evidence.bin", payload),
            {"spear_repo": repo},
            role="fixture",
            owner="undeclared mount test",
        )


def test_artifact_authentication_rejects_nested_escape_from_tmp_mount(tmp_path):
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    repo.mkdir()
    workspace.mkdir()
    outside.mkdir()
    (repo / "tmp").symlink_to(workspace, target_is_directory=True)
    (workspace / "nested").symlink_to(outside, target_is_directory=True)
    payload = b"nested outside artifact\n"
    (outside / "evidence.bin").write_bytes(payload)

    with pytest.raises(
        contracts.ContractError,
        match="artifact escapes trusted mount spear_repo:tmp",
    ):
        input_builder.authenticate_artifact_record(
            _artifact_record("spear_repo", "tmp/nested/evidence.bin", payload),
            {"spear_repo": repo},
            role="fixture",
            owner="nested escape test",
        )


def static_object_profile(measurement_artifact: dict) -> dict:
    return {
        "schema": contracts.PROFILE_SCHEMA,
        "profile_schema_id": "appliance_alarm_clock_v1",
        "profile_revision": "2026_07_27_v1",
        "asset_class": "static_object",
        "lineage_group_id": "alarm_clock_reference_01",
        "state_classification": "research_candidate",
        "taxonomy": {"category": "appliance", "object_type": "alarm_clock"},
        "base_template": {
            "template_id": "alarm_clock_reference_01",
            "kind": "text_prompt_only",
            "artifact": None,
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": {"style": "twin_bell_analog", "material": "metal"},
        "sampled_attribute_domains": {"body_color": ["black", "white"]},
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "flux2_pixal3d_static_v1",
            "prompt_template_id": "static_object_t2i_v1",
            "positive_template": (
                "A {body_color} {style} {material} {object_type}, a single "
                "household {category} in a clean product photo."
            ),
            "pose_guard_prompt": (
                "Single centered object, three-quarter product view, level "
                "camera, every part fully visible, plain background."
            ),
            "negative_prompt": (
                "cropped object, multiple objects, human hands, text, "
                "background clutter"
            ),
            "value_labels": {
                "category": {"appliance": "appliance"},
                "object_type": {"alarm_clock": "alarm clock"},
                "style": {"twin_bell_analog": "twin-bell analog"},
                "material": {"metal": "metal"},
                "body_color": {"black": "matte black", "white": "white"},
            },
            "model_revisions": {
                "flux2": animal_worker.MODEL_REVISION,
                "pixal3d": "pixal_revision",
                "dino": "dino_revision",
            },
            "base_acquisition_policy": one_shot.static_base_acquisition_record(),
        },
        "target_physical_profiles": {
            "profile_id": "alarm_clock_physical_v1",
            "control_attribute": None,
            "measurement": "height_cm",
            "mode": "absolute_measurement",
            "reference_value_cm": 13.0,
            "reference_provenance": {
                "status": "verified",
                "source_id": "fixture_product_measurement_v1",
                "artifact": measurement_artifact,
                "notes": "Test fixture only.",
            },
            "values": {"fixed": {"target_value_cm": 13.0, "tolerance_cm": 1.5}},
        },
        "rig_profile": None,
        "acoustic_profile": {
            "profile_id": "alarm_clock_ring_v1",
            "default_event_class": "alarm_clock_ring",
            "allowed_event_classes": ["alarm_clock_ring", "silent"],
            "selection_attributes": ["object_type", "style"],
        },
        "locked_attributes": ["category", "object_type", "style", "material"],
        "qa_contract": {
            "subject_label": "alarm clock",
            "attributes": {
                "body_color": {
                    "kind": "categorical",
                    "label": "body color",
                    "value_labels": {"black": "matte black", "white": "white"},
                    "identification_question": (
                        "What is the body color of {instance_label}?"
                    ),
                }
            },
        },
    }


def _published_static_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    payload = b'{"height_cm": 13.0}\n'
    measurement_path = tmp_path / "references/alarm_clock_measurement.json"
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.write_bytes(payload)
    measurement_artifact = {
        "root_id": "fixture_root",
        "path": "references/alarm_clock_measurement.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    profile = contracts.validate_attribute_profile(
        static_object_profile(measurement_artifact)
    )
    artifact_roots = {"fixture_root": tmp_path}
    authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, artifact_roots
        )
    }
    files = input_builder.compile_inputs(
        profiles=[profile],
        count_per_profile=2,
        seed=20260727,
        plan_id="static_object_preflight_canary_v1",
        split_salt="static-object-preflight-v1",
        max_qa_pairs_per_split=None,
        artifact_authentication=authentication,
    )
    bundle_dir = tmp_path / "bundle"
    input_builder.publish_output(bundle_dir, files)
    return bundle_dir, artifact_roots


def test_preflight_builds_static_object_jobs_without_reference_images(tmp_path):
    bundle_dir, artifact_roots = _published_static_bundle(tmp_path)

    preflight = execution.build_execution_preflight(bundle_dir, artifact_roots)

    assert set(preflight["routes"]) == set(contracts.ROUTES)
    static_jobs = preflight["routes"]["flux2_pixal3d_static_v1"]
    assert len(static_jobs) == 2
    assert preflight["execution_summary"]["static_object_job_count"] == 2
    assert preflight["execution_summary"]["animal_job_count"] == 0
    for job in static_jobs:
        assert "reference" not in job
        assert job["rig_profile"] is None
        plan = job["generation_plan"]
        assert plan["schema"] == "flux2_pixal3d_static_generation_plan_v1"
        assert plan["base_template"]["kind"] == "text_prompt_only"
        assert plan["base_template"]["artifact"] is None
        one_shot.validate_flux_job(job)
    assert execution.validate_execution_preflight(preflight) == preflight


def test_static_worker_partition_validates_and_rejects_injected_reference(tmp_path):
    bundle_dir, artifact_roots = _published_static_bundle(tmp_path)
    preflight = execution.build_execution_preflight(bundle_dir, artifact_roots)
    jobs = json.loads(
        json.dumps(preflight["routes"]["flux2_pixal3d_static_v1"])
    )
    partition = {
        "schema": animal_worker.PARTITION_SCHEMA,
        "execution_preflight_sha256": preflight["preflight_sha256"],
        "one_shot_execution": one_shot.stage_record("flux2"),
        "model": animal_flux.MODEL,
        "parameters": animal_flux.PARAMETERS,
        "jobs": jobs,
    }
    partition["partition_sha256"] = animal_flux._json_sha256(partition)

    assert animal_worker.validate_partition(partition) == partition

    tampered = json.loads(json.dumps(partition))
    tampered["jobs"][0]["reference"] = {
        "root_id": "fixture_root",
        "path": "references/alarm_clock_measurement.json",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "resolved_path": str(tmp_path / "references/alarm_clock_measurement.json"),
    }
    tampered["partition_sha256"] = animal_flux._json_sha256(
        {key: value for key, value in tampered.items() if key != "partition_sha256"}
    )
    with pytest.raises(ValueError, match="static job contract"):
        animal_worker.validate_partition(tampered)


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
        "static_object_job_count": 0,
        "stable_animal_job_count": 0,
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
    jobs = json.loads(json.dumps(jobs))
    for job in jobs:
        job["generation_plan"]["base_acquisition_policy"] = (
            one_shot.base_acquisition_record()
        )
    partition = {
        "schema": animal_worker.PARTITION_SCHEMA,
        "execution_preflight_sha256": preflight["preflight_sha256"],
        "one_shot_execution": one_shot.stage_record("flux2"),
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
