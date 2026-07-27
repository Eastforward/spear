from __future__ import annotations

import copy
import os
import stat
import subprocess
from pathlib import Path

from PIL import Image
import pytest

from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as preparation


REQUEST_SHA256 = "1" * 64
PROFILE_SHA256 = "2" * 64
PREFLIGHT_SHA256 = "3" * 64
BATCH_SHA256 = "4" * 64
INSTANCE_ID = "kitchen_appliance_microwave_test_0001"
PROFILE_ID = "kitchen_appliance_microwave_test_v1"


def _execution_job(route: str) -> dict:
    static = route == "flux2_pixal3d_static_v1"
    prefix = "static" if static else "animal"
    return {
        "execution_job_id": f"{prefix}_{REQUEST_SHA256[:16]}",
        "profile_schema_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "sampled_attributes": {"body_color": "white"},
        "consumer_requests": [
            {
                "instance_id": INSTANCE_ID,
                "request_sha256": REQUEST_SHA256,
            }
        ],
        "generation_plan": {
            "schema": (
                "flux2_pixal3d_static_generation_plan_v1"
                if static
                else "flux2_pixal3d_generation_plan_v1"
            ),
            "route": route,
            "generation_seed": 27,
            "model_revisions": {
                "pixal3d": preparation.PIXAL_MODEL_REVISION,
                "dino": preparation.DINO_REVISION,
            },
            "base_template": {
                "kind": "text_prompt_only" if static else "reference_image",
                "artifact": None if static else {"sha256": "5" * 64},
            },
            "base_acquisition_policy": (
                one_shot.static_base_acquisition_record()
                if static
                else one_shot.base_acquisition_record()
            ),
        },
        "target_physical_profile": {
            "measurement": "height_cm" if static else "shoulder_height_cm",
            "target_value_cm": 28.0,
        },
        "rig_profile": (
            None
            if static
            else {
                "profile_id": "quadruped_test_v1",
                "skeleton_family": "test",
                "actions": ["Walking", "Idle"],
                "front_axis": "positive_x",
            }
        ),
    }


def _candidate(job: dict, *, static: bool) -> dict:
    index = {
        "instance_id": INSTANCE_ID,
        "execution_job_id": job["execution_job_id"],
        "profile_schema_id": PROFILE_ID,
        "sampled_attributes": {"body_color": "white"},
        "candidate": {
            "path": "candidate.png",
            "sha256": "6" * 64,
            "size_bytes": 1,
        },
    }
    files: dict[str, object] = {"candidate": Path("candidate.png")}
    if not static:
        index["source"] = {
            "path": "source.png",
            "sha256": "7" * 64,
            "size_bytes": 1,
        }
        files["source"] = Path("source.png")
    return {
        "index": index,
        "manifest": {
            "instance_id": INSTANCE_ID,
            "execution_job_id": job["execution_job_id"],
            "profile_schema_id": PROFILE_ID,
            "profile_sha256": PROFILE_SHA256,
            "request_sha256": REQUEST_SHA256,
            "sampled_attributes": {"body_color": "white"},
            "input": None if static else {"sha256": "7" * 64},
            "one_shot_execution": one_shot.stage_record("flux2"),
        },
        "files": files,
    }


def _route_fixture(route: str, *, declare_route: bool = True):
    job = _execution_job(route)
    static = route == "flux2_pixal3d_static_v1"
    selection = {"semantics": "predeclared_request_subset_only_not_output_ranking"}
    if declare_route:
        selection["route"] = route
    flux_batch = {"selection": selection}
    preflight = {
        "routes": {
            "flux2_pixal3d_animal_v1": [] if static else [job],
            "flux2_pixal3d_static_v1": [job] if static else [],
        }
    }
    candidates = {INSTANCE_ID: _candidate(job, static=static)}
    return flux_batch, preflight, candidates, job


def test_static_route_compiles_an_explicit_unrigged_controlled_request():
    flux_batch, preflight, candidates, job = _route_fixture(
        "flux2_pixal3d_static_v1"
    )

    route, route_contract, jobs = preparation._authenticated_route_jobs(
        flux_batch, preflight, candidates
    )
    request = preparation._controlled_request(
        jobs[INSTANCE_ID],
        instance_id=INSTANCE_ID,
        route=route,
        route_contract=route_contract,
    )

    assert route == "flux2_pixal3d_static_v1"
    assert route_contract["asset_class"] == "static_object"
    assert "rig_mode" not in route_contract
    assert request["asset_class"] == "static_object"
    assert request["route"] == route
    assert request["profile_sha256"] == PROFILE_SHA256
    assert request["rig_profile"] is None
    assert job["generation_plan"]["base_template"]["artifact"] is None


def test_route_tampering_is_rejected_against_authenticated_static_job():
    flux_batch, preflight, candidates, _job = _route_fixture(
        "flux2_pixal3d_static_v1"
    )
    flux_batch["selection"]["route"] = "flux2_pixal3d_animal_v1"

    with pytest.raises(
        contracts.ContractError,
        match="route differs from its authenticated execution jobs",
    ):
        preparation._authenticated_route_jobs(flux_batch, preflight, candidates)


def test_profile_tampering_is_rejected_against_authenticated_job():
    flux_batch, preflight, candidates, _job = _route_fixture(
        "flux2_pixal3d_static_v1"
    )
    candidates[INSTANCE_ID]["manifest"]["profile_sha256"] = "f" * 64

    with pytest.raises(
        contracts.ContractError,
        match="manifest profile_sha256",
    ):
        preparation._authenticated_route_jobs(flux_batch, preflight, candidates)


def test_legacy_animal_route_remains_compatible_and_animated():
    flux_batch, preflight, candidates, _job = _route_fixture(
        "flux2_pixal3d_animal_v1", declare_route=False
    )

    route, route_contract, jobs = preparation._authenticated_route_jobs(
        flux_batch, preflight, candidates
    )
    request = preparation._controlled_request(
        jobs[INSTANCE_ID],
        instance_id=INSTANCE_ID,
        route=route,
        route_contract=route_contract,
    )

    assert route == "flux2_pixal3d_animal_v1"
    assert route_contract["asset_class"] == "animal"
    assert route_contract["rig_mode"] == "animated_transfer"
    assert request["asset_class"] == "animal"
    assert set(request["rig_profile"]["actions"]) == {"Walking", "Idle"}


@pytest.mark.parametrize(
    ("route", "asset_class", "rig_mode"),
    [
        ("flux2_pixal3d_static_v1", "static_object", None),
        ("flux2_pixal3d_animal_v1", "animal", "animated_transfer"),
    ],
)
def test_prepare_pixal_inputs_publishes_the_authenticated_route_contract(
    tmp_path, monkeypatch, route, asset_class, rig_mode
):
    static = asset_class == "static_object"
    flux_batch, preflight, candidates, _job = _route_fixture(route)
    candidate_path = tmp_path / "candidate.png"
    Image.new("RGB", (1024, 1024), (220, 220, 220)).save(candidate_path)
    candidate_record = {
        "path": "candidate.png",
        "sha256": preparation._sha256_file(candidate_path),
        "size_bytes": candidate_path.stat().st_size,
    }
    candidates[INSTANCE_ID]["index"]["candidate"] = candidate_record
    candidates[INSTANCE_ID]["files"]["candidate"] = candidate_path

    preflight_path = tmp_path / "execution_preflight.json"
    preflight_path.write_text("{}\n", encoding="utf-8")
    preflight["preflight_sha256"] = PREFLIGHT_SHA256
    flux_batch.update(
        {
            "batch_sha256": BATCH_SHA256,
            "execution_preflight": {
                "path": str(preflight_path),
                "sha256": preparation._sha256_file(preflight_path),
                "preflight_sha256": PREFLIGHT_SHA256,
            },
            "one_shot_execution": one_shot.stage_record("flux2"),
        }
    )

    review_payload = {
        "schema": (
            preparation.review.STATIC_REVIEW_SCHEMA
            if static
            else preparation.review.REVIEW_SCHEMA
        ),
        "instance_id": INSTANCE_ID,
        "candidate": {"sha256": candidate_record["sha256"]},
        "decision": "approved_for_pixal3d",
    }
    review_payload["review_sha256"] = preparation._hash_without(
        review_payload, "review_sha256"
    )
    review_path = tmp_path / "review.json"
    contracts.write_json_no_replace(review_path, review_payload)
    review_record = {
        "path": review_path.name,
        "sha256": preparation._sha256_file(review_path),
        "size_bytes": review_path.stat().st_size,
    }
    review_batch_path = tmp_path / "review_batch.json"
    review_batch_path.write_text("{}\n", encoding="utf-8")
    review_batch = {
        "schema": (
            preparation.review.STATIC_BATCH_REVIEW_SCHEMA
            if static
            else preparation.review.BATCH_REVIEW_SCHEMA
        ),
        "review_domain": "static_object" if static else "animal",
        "flux2_batch": {
            "path": str(tmp_path / "flux2_batch.json"),
            "batch_sha256": BATCH_SHA256,
        },
        "reviews": [
            {
                "instance_id": INSTANCE_ID,
                "candidate_sha256": candidate_record["sha256"],
                "review": review_record,
            }
        ],
        "approved_count": 1,
        "review_batch_sha256": "8" * 64,
    }

    monkeypatch.setattr(
        preparation, "load_review_batch", lambda _path: copy.deepcopy(review_batch)
    )
    monkeypatch.setattr(
        preparation.review,
        "load_flux_batch",
        lambda _path: (tmp_path, copy.deepcopy(flux_batch), candidates),
    )
    monkeypatch.setattr(
        preparation.material_execution,
        "_load_preflight",
        lambda _path: copy.deepcopy(preflight),
    )
    monkeypatch.setattr(
        preparation.material_execution.native,
        "_seal_readonly_tree",
        lambda _path: None,
    )
    reauthentication_calls = []
    original_reauthenticate = preparation._reauthenticate_isnet_execution

    def tracked_reauthentication(execution):
        original_reauthenticate(execution)
        reauthentication_calls.append(str(execution["staged_worker"]))

    monkeypatch.setattr(
        preparation,
        "_reauthenticate_isnet_execution",
        tracked_reauthentication,
    )
    executed_commands = []
    rename_calls = []
    original_rename_noreplace = preparation._rename_noreplace

    def tracked_rename_noreplace(source, destination):
        rename_calls.append((str(source), str(destination)))
        original_rename_noreplace(source, destination)

    monkeypatch.setattr(
        preparation, "_rename_noreplace", tracked_rename_noreplace
    )

    def fake_isnet(command, **_kwargs):
        executed_commands.append(list(command))
        assert command[0] == str(preparation.ISNET_PYTHON.resolve(strict=True))
        assert Path(command[1]).name == "controlled_animal_isnet_worker.py"
        assert Path(command[1]).parent.name == ".runtime_commands"
        assert Path(command[1]) != preparation.ISNET_WORKER
        assert stat.S_IMODE(Path(command[1]).parent.stat().st_mode) == 0o555
        replacement = tmp_path / "replacement_isnet_worker.py"
        replacement.write_bytes(b"print('replacement worker')\n")
        with pytest.raises(PermissionError):
            os.replace(replacement, Path(command[1]))
        assert replacement.is_file()
        jobs_path = Path(command[command.index("--jobs") + 1])
        status_path = Path(command[command.index("--status") + 1])
        jobs = contracts.load_json(jobs_path)["jobs"]
        status_jobs = []
        for job in jobs:
            alpha = Path(job["alpha_path"])
            rgba = Path(job["rgba_path"])
            alpha.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (1024, 1024), 255).save(alpha)
            Image.new("RGBA", (1024, 1024), (220, 220, 220, 255)).save(rgba)
            status_jobs.append(
                {
                    "instance_id": job["instance_id"],
                    "alpha_sha256": preparation._sha256_file(alpha),
                    "rgba_sha256": preparation._sha256_file(rgba),
                    "foreground_fraction_at_128": 0.5,
                    "foreground_bbox_xyxy": [0, 0, 1023, 1023],
                }
            )
        contracts.write_json_no_replace(
            status_path,
            {
                "schema": preparation.isnet.STATUS_SCHEMA,
                "status": "passed",
                "model": {
                    "path": str(preparation.isnet.MODEL_PATH),
                    "sha256": preparation.isnet.MODEL_SHA256,
                    "name": "isnet-general-use",
                },
                "passed_count": len(jobs),
                "failed_count": 0,
                "jobs": status_jobs,
            },
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(preparation.subprocess, "run", fake_isnet)

    output_root = tmp_path / "pixal_inputs"
    manifest_path = preparation.prepare_pixal_inputs(
        review_batch_path,
        output_root,
        tmp_path / "pixal_outputs",
    )
    manifest = contracts.load_json(manifest_path)

    assert manifest["asset_class"] == asset_class
    assert manifest["route"] == route
    assert manifest["job_count"] == 1
    assert len(reauthentication_calls) == 3
    assert len(executed_commands) == 1
    assert len(rename_calls) == 1
    assert rename_calls[0][1] == str(output_root)
    assert manifest["automatic_checks"][
        "static_jobs_have_no_rig_or_animation_binding"
    ] is True
    assert manifest["automatic_checks"][
        "isnet_runtime_inputs_reauthenticated_before_and_after_execution"
    ] is True
    assert manifest["automatic_checks"][
        "isnet_worker_executed_from_frozen_published_copy"
    ] is True
    isnet_receipt = manifest["isnet"]
    assert set(isnet_receipt) == {
        "schema",
        "model",
        "python",
        "worker",
        "jobs",
        "working_directory",
        "command",
        "command_sha256",
        "executed_command",
        "executed_command_sha256",
        "path_rebinding",
        "status",
        "log",
    }
    assert (
        isnet_receipt["schema"] == preparation.ISNET_EXECUTION_RECEIPT_SCHEMA
    )
    for record in (
        isnet_receipt["model"],
        isnet_receipt["python"]["configured"],
        isnet_receipt["python"]["resolved"],
        isnet_receipt["worker"]["source"],
        isnet_receipt["worker"]["executed"],
        isnet_receipt["jobs"],
    ):
        assert set(record) == {"path", "sha256", "size_bytes"}
        path = Path(record["path"])
        assert path.is_file()
        assert preparation._sha256_file(path) == record["sha256"]
        assert path.stat().st_size == record["size_bytes"]
    assert isnet_receipt["python"]["configured"]["path"] == str(
        preparation.ISNET_PYTHON
    )
    assert isnet_receipt["python"]["resolved"]["path"] == str(
        preparation.ISNET_PYTHON.resolve(strict=True)
    )
    assert isnet_receipt["worker"]["source"]["path"] == str(
        preparation.ISNET_WORKER
    )
    assert Path(isnet_receipt["worker"]["executed"]["path"]).is_relative_to(
        output_root
    )
    assert isnet_receipt["command"][1] == isnet_receipt["worker"]["executed"][
        "path"
    ]
    assert isnet_receipt["command"][3] == isnet_receipt["jobs"]["path"]
    assert isnet_receipt["command_sha256"] == preparation._json_sha256(
        isnet_receipt["command"]
    )
    assert isnet_receipt[
        "executed_command_sha256"
    ] == preparation._json_sha256(isnet_receipt["executed_command"])
    assert isnet_receipt["executed_command"] == executed_commands[0]
    assert isnet_receipt["executed_command"] == preparation._rebind_command_root(
        isnet_receipt["command"],
        source_root=Path(isnet_receipt["path_rebinding"]["published_root"]),
        destination_root=Path(isnet_receipt["path_rebinding"]["staging_root"]),
    )
    assert isnet_receipt["path_rebinding"]["published_root"] == str(output_root)
    job = manifest["jobs"][0]
    assert job["asset_class"] == asset_class
    assert job["route"] == route
    if static:
        assert "rig_mode" not in job
        assert job["controlled_request"]["rig_profile"] is None
        assert "animated_transfer" not in contracts.canonical_json(manifest)
    else:
        assert job["rig_mode"] == rig_mode
        assert set(job["controlled_request"]["rig_profile"]["actions"]) == {
            "Walking",
            "Idle",
        }
    assert job["controlled_request"]["profile_sha256"] == PROFILE_SHA256


def _isolated_isnet_execution(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    resolved_python = runtime_root / "python3.10"
    resolved_python.write_bytes(b"#!/bin/sh\nexit 0\n")
    resolved_python.chmod(0o755)
    configured_python = runtime_root / "python"
    configured_python.symlink_to(resolved_python.name)
    worker = runtime_root / "controlled_animal_isnet_worker.py"
    worker.write_bytes(b"print('fixture worker')\n")
    model = runtime_root / "isnet-general-use.onnx"
    model.write_bytes(b"fixture model")
    monkeypatch.setattr(preparation, "ISNET_PYTHON", configured_python)
    monkeypatch.setattr(preparation, "ISNET_WORKER", worker)
    monkeypatch.setattr(preparation.isnet, "MODEL_PATH", model)
    monkeypatch.setattr(
        preparation.isnet, "MODEL_SHA256", preparation._sha256_file(model)
    )
    staging = tmp_path / ".pixal_inputs.fixture.staging"
    staging.mkdir()
    jobs_path = staging / "isnet_jobs.json"
    contracts.write_json_no_replace(
        jobs_path,
        {"schema": preparation.isnet.JOBS_SCHEMA, "jobs": [{"fixture": True}]},
    )
    output_root = tmp_path / "pixal_inputs"
    execution = preparation._prepare_isnet_execution(
        staging=staging,
        output_root=output_root,
        jobs_path=jobs_path,
        status_path=staging / "isnet_status.json",
    )
    return execution, {
        "configured_python": configured_python,
        "resolved_python": resolved_python,
        "worker": worker,
        "model": model,
        "jobs": jobs_path,
    }


@pytest.mark.parametrize(
    ("tamper_target", "message"),
    [
        ("python_target", "configured ISNet Python target changed"),
        ("worker_source", "ISNet worker source changed"),
        ("frozen_worker", "frozen ISNet worker changed"),
        ("model", "pinned ISNet model changed"),
        ("jobs", "ISNet jobs changed"),
        ("command", "ISNet command execution receipt changed"),
        ("runtime_directory", "frozen ISNet worker directory changed"),
    ],
)
def test_isnet_execution_reauthentication_fails_closed_on_runtime_tampering(
    tmp_path, monkeypatch, tamper_target, message
):
    execution, paths = _isolated_isnet_execution(tmp_path, monkeypatch)
    preparation._reauthenticate_isnet_execution(execution)

    if tamper_target == "python_target":
        replacement = paths["resolved_python"].with_name("python3.10.same-bytes")
        replacement.write_bytes(paths["resolved_python"].read_bytes())
        replacement.chmod(0o755)
        paths["configured_python"].unlink()
        paths["configured_python"].symlink_to(replacement.name)
    elif tamper_target == "worker_source":
        paths["worker"].write_bytes(b"print('changed worker')\n")
    elif tamper_target == "frozen_worker":
        frozen = Path(execution["staged_worker"])
        frozen.chmod(0o644)
        frozen.write_bytes(b"print('changed frozen worker')\n")
    elif tamper_target == "model":
        paths["model"].write_bytes(b"changed model")
    elif tamper_target == "jobs":
        paths["jobs"].write_bytes(b'{"changed":true}\n')
    elif tamper_target == "runtime_directory":
        Path(execution["staged_worker"]).parent.chmod(0o755)
    else:
        execution["receipt"]["command"].append("--changed")

    with pytest.raises(contracts.ContractError, match=message):
        preparation._reauthenticate_isnet_execution(execution)


def test_atomic_no_replace_rejects_a_concurrently_created_empty_output_root(
    tmp_path,
):
    staging = tmp_path / ".pixal_inputs.fixture.staging"
    staging.mkdir()
    staged_file = staging / "pixal_inputs_manifest.json"
    staged_file.write_bytes(b'{"fixture":true}\n')
    output_root = tmp_path / "pixal_inputs"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="concurrently-created"):
        preparation._rename_noreplace(staging, output_root)

    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []
    assert staged_file.read_bytes() == b'{"fixture":true}\n'
