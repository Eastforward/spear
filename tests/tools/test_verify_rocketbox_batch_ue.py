import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.verify_rocketbox_batch_ue import (
    aggregate_verification_results,
    build_command,
    build_jobs,
    run_command_with_timeout,
    select_jobs_for_resume,
    validate_completed_job,
)


def _inventory(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "rocketbox_human_inventory_v1",
                "population": {"total": 115},
                "automatic_checks": {"overall": "passed"},
                "avatars": [
                    {
                        "base_avatar_id": "rocketbox_children_female_child_01",
                        "inventory_status": "passed",
                    },
                    {
                        "base_avatar_id": "rocketbox_adults_male_adult_01",
                        "inventory_status": "passed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _materialize_job_inputs(normalized_root: Path, manifest_root: Path) -> None:
    for avatar_id in (
        "rocketbox_adults_male_adult_01",
        "rocketbox_children_female_child_01",
    ):
        source = normalized_root / f"{avatar_id}_original_ue_v1"
        source.mkdir(parents=True)
        (source / "runtime.glb").write_bytes(b"glb")
        (source / "normalization_manifest.json").write_text("{}")
        target = manifest_root / f"{avatar_id}_original_ue_v1"
        target.mkdir(parents=True)
        (target / "ue_import_manifest.json").write_text("{}")


def test_builds_sorted_direct_process_jobs_with_isolated_logs(tmp_path):
    inventory = _inventory(tmp_path)
    normalized_root = tmp_path / "normalized"
    manifest_root = tmp_path / "manifests"
    log_root = tmp_path / "logs"
    _materialize_job_inputs(normalized_root, manifest_root)

    jobs = build_jobs(inventory, normalized_root, manifest_root, log_root)

    assert [job.base_avatar_id for job in jobs] == [
        "rocketbox_adults_male_adult_01",
        "rocketbox_children_female_child_01",
    ]
    assert jobs[0].log_path != jobs[1].log_path
    assert jobs[0].environment["ROCKETBOX_NATIVE_VERIFY_ONLY"] == "1"
    assert jobs[0].environment["ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH"] == "1"


def test_command_runs_gate_directly_and_disables_shared_registry_writes(tmp_path):
    inventory = _inventory(tmp_path)
    normalized_root = tmp_path / "normalized"
    manifest_root = tmp_path / "manifests"
    log_root = tmp_path / "logs"
    _materialize_job_inputs(normalized_root, manifest_root)
    job = build_jobs(inventory, normalized_root, manifest_root, log_root)[0]

    command = build_command(
        job,
        ue_editor=Path("/opt/UnrealEditor"),
        project=Path("/repo/SpearSim.uproject"),
        gate_script=Path("/repo/import_gate_rocketbox_native_editor.py"),
    )

    assert command[0] == "/opt/UnrealEditor"
    assert "-RenderOffscreen" in command
    assert "-NoAssetRegistryCacheWrite" in command
    assert f"-AbsLog={job.log_path}" in command
    assert command[-1] == "-script=/repo/import_gate_rocketbox_native_editor.py"
    assert all("import_rocketbox_batch_editor.py" not in item for item in command)


def test_completed_job_requires_hash_bound_reload_pass(tmp_path):
    inventory = _inventory(tmp_path)
    normalized_root = tmp_path / "normalized"
    manifest_root = tmp_path / "manifests"
    log_root = tmp_path / "logs"
    avatar_id = "rocketbox_adults_male_adult_01"
    tag = f"{avatar_id}_original_ue_v1"
    _materialize_job_inputs(normalized_root, manifest_root)
    target = manifest_root / tag
    manifest = target / "ue_import_manifest.json"
    manifest.write_text("{}")
    job = build_jobs(inventory, normalized_root, manifest_root, log_root)[0]

    manifest.write_text(
        json.dumps(
            {
                "schema": "rocketbox_batch_native_ue_import_v1",
                "base_avatar_id": avatar_id,
                "tag": tag,
                "reload_verification": {"status": "passed"},
                "runtime_contract": {
                    "bone_count": 80,
                    "actor_scale": 1,
                    "bounds": {
                        "height_passed": True,
                        "authored_height_preserved": True,
                        "ground_passed": True,
                    },
                },
            }
        )
    )
    validate_completed_job(job)

    payload = json.loads(manifest.read_text())
    payload["reload_verification"]["status"] = "pending"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="reload"):
        validate_completed_job(job)


def test_timeout_kills_child_processes_that_inherit_console_output(tmp_path):
    console_log = tmp_path / "console.log"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        ),
    ]
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="timed out"):
        run_command_with_timeout(
            command,
            cwd=tmp_path,
            environment=os.environ.copy(),
            console_log=console_log,
            timeout_seconds=0.2,
        )

    assert time.monotonic() - started < 5
    assert console_log.is_file()


def test_resume_selects_only_manifests_without_passed_reload(tmp_path):
    inventory = _inventory(tmp_path)
    normalized_root = tmp_path / "normalized"
    manifest_root = tmp_path / "manifests"
    log_root = tmp_path / "logs"
    _materialize_job_inputs(normalized_root, manifest_root)
    jobs = build_jobs(inventory, normalized_root, manifest_root, log_root)
    passed = jobs[0]
    pending = jobs[1]
    passed.ue_manifest.write_text(
        json.dumps({"reload_verification": {"status": "passed"}})
    )
    pending.ue_manifest.write_text(
        json.dumps({"reload_verification": {"status": "pending"}})
    )

    selected = select_jobs_for_resume(jobs)

    assert selected == [pending]

    selected_with_process_failure = select_jobs_for_resume(
        jobs, failed_avatar_ids={passed.base_avatar_id}
    )
    assert selected_with_process_failure == [passed, pending]


def test_current_process_failure_overrides_an_old_passed_manifest(tmp_path):
    inventory = _inventory(tmp_path)
    normalized_root = tmp_path / "normalized"
    manifest_root = tmp_path / "manifests"
    log_root = tmp_path / "logs"
    _materialize_job_inputs(normalized_root, manifest_root)
    jobs = build_jobs(inventory, normalized_root, manifest_root, log_root)
    failed = jobs[0]
    passed = jobs[1]
    for job in jobs:
        job.ue_manifest.write_text(
            json.dumps(
                {
                    "schema": "rocketbox_batch_native_ue_import_v1",
                    "base_avatar_id": job.base_avatar_id,
                    "tag": job.tag,
                    "reload_verification": {"status": "passed"},
                    "runtime_contract": {
                        "bone_count": 80,
                        "actor_scale": 1,
                        "bounds": {
                            "height_cm": 175,
                            "authored_height_cm": 175,
                            "height_passed": True,
                            "authored_height_preserved": True,
                            "ground_passed": True,
                        },
                    },
                }
            )
        )

    results, failures = aggregate_verification_results(
        jobs,
        current_results={passed.base_avatar_id: {"base_avatar_id": passed.base_avatar_id}},
        current_failures={
            failed.base_avatar_id: {
                "base_avatar_id": failed.base_avatar_id,
                "error": "process returned 1",
            }
        },
    )

    assert [item["base_avatar_id"] for item in failures] == [
        failed.base_avatar_id
    ]
    assert failed.base_avatar_id not in {
        item["base_avatar_id"] for item in results
    }
