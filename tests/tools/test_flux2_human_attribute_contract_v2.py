from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

from tools import flux2_edit_human_attributes as runner


SPEAR_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_JOBS = SPEAR_ROOT / "tmp/human_attribute_instances_v1/jobs_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_file_rejects_symlink_instead_of_following_it(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"authenticated")
    link = tmp_path / "link.bin"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="without following links"):
        runner.sha256_file(link)


def test_production_jobs_v2_bind_every_reproducibility_and_downstream_field():
    jobs = runner.load_jobs(PRODUCTION_JOBS)

    assert [job["case_id"] for job in jobs] == list(runner.CASE_CONTRACTS)
    for job in jobs:
        assert job["downstream_asset_id"] == f"route2_{job['case_id']}_v1"
        assert job["base_qualified_candidate"].endswith(
            f"/{job['base_asset_id']}/qualified_candidate_v1.json"
        )
        assert job["mask_construction_version"] == "human_attribute_source_semantics_v2"
        assert job["target_parameters"] == runner.CASE_CONTRACTS[job["case_id"]][
            "target_parameters"
        ]
        assert job["model"] == {
            "name": "black-forest-labs/FLUX.2-klein-4B",
            "root": str(runner.MODEL_ROOT),
            "revision": runner.MODEL_REVISION,
            "inventory": str(runner.MODEL_INVENTORY_PATH),
            "inventory_sha256": runner.MODEL_INVENTORY_SHA256,
            "local_files_only": True,
            "max_sequence_length": 512,
        }
        assert job["runner_sha256"] == runner.sha256_file(runner.RUNNER_PATH)
        assert set(job["mask_bundle"]) == {"path", "sha256"}
        assert set(job["mask_decision"]) == {"path", "sha256"}
        assert set(job["source_alpha"]) == {"path", "sha256"}
        assert set(job["source_rgba"]) == {"path", "sha256"}
        assert set(job["source_candidate_manifest"]) == {"path", "sha256"}
        assert job["isnet"]["model_path"] == str(runner.ISNET_MODEL_PATH)
        assert job["isnet"]["model_sha256"] == runner.ISNET_MODEL_SHA256


def test_production_source_and_mask_preflight_authenticates_recursive_chain_without_gpu():
    preflight = runner.preflight_batch(
        PRODUCTION_JOBS,
        SPEAR_ROOT / "tmp/human_attribute_instances_v1/candidates_v1",
        authenticate_model=False,
        require_base_qa=False,
    )

    assert preflight["jobs_sha256"] == _sha(PRODUCTION_JOBS)
    assert len(preflight["jobs"]) == 7
    assert preflight["execution_authorized"] is False
    for record in preflight["jobs"]:
        assert record["source"]["candidate_manifest"]["sha256"]
        assert record["source"]["source_alpha"]["sha256"]
        assert record["source"]["source_rgba"]["sha256"]
        assert set(record["mask_bundle"]["assets"]) == {
            "edit_core.png",
            "transition_band.png",
            "protected_guard.png",
            "overlay.png",
        }
        assert record["mask_bundle"]["partition_exact"] is True
        assert record["mask_bundle"]["agent_decision"]["status"] == (
            "agent_qa_passed_pending_user_acceptance"
        )


def test_execution_preflight_fails_closed_until_both_base_route2_reviews_pass():
    with pytest.raises(ValueError, match="base Route-2 qualification"):
        runner.preflight_batch(
            PRODUCTION_JOBS,
            SPEAR_ROOT / "tmp/human_attribute_instances_v1/candidates_v1",
            authenticate_model=False,
            require_base_qa=True,
        )


def test_base_qualification_consumes_the_canonical_qualified_candidate_pointer(
    monkeypatch, tmp_path
):
    base_id = "rocketbox_male_adult_01"
    route_root = tmp_path / "pixal_tokenrig_route2_v1"
    pointer = route_root / base_id / "qualified_candidate_v1.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}", encoding="utf-8")
    pointer.chmod(0o444)
    review_dir = pointer.parent / "fitted_skeleton_v1/sanitized_weights_v1/dynamic_review_v1"
    review_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROUTE2_OUTPUT_ROOT", route_root)
    monkeypatch.setattr(
        runner.qualified_candidate,
        "validate_qualified_candidate",
        lambda path: {
            "asset_id": base_id,
            "base_avatar_id": base_id,
            "status": "agent_qa_passed_pending_user_acceptance",
            "final_branch": {
                "branch_id": "sanitized_weights",
                "path": str(review_dir.parent),
                "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
            },
            "dynamic": {"review_dir": str(review_dir)},
        },
    )
    job = copy.deepcopy(json.loads(PRODUCTION_JOBS.read_text())["jobs"][0])
    job["runner_sha256"] = runner.sha256_file(runner.RUNNER_PATH)
    job["base_qualified_candidate"] = str(pointer)

    result = runner.authenticate_base_route2_qualification(job)

    assert result == {
        "asset_id": base_id,
        "status": "agent_qa_passed_pending_user_acceptance",
        "qualified_candidate": {
            "path": str(pointer),
            "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
            "size_bytes": pointer.stat().st_size,
        },
        "final_branch": {
            "branch_id": "sanitized_weights",
            "path": str(review_dir.parent),
            "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
        },
        "review_dir": str(review_dir),
    }


def test_base_qa_without_model_authentication_never_authorizes_execution(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner,
        "authenticate_base_route2_qualification",
        lambda job: {
            "asset_id": job["base_asset_id"],
            "review_dir": "/qualified/dynamic_review_v1",
            "qualified_candidate": {
                "path": job["base_qualified_candidate"],
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "final_branch": {
                "branch_id": "direct",
                "path": "/qualified",
                "relative_root": ".",
            },
            "status": "agent_qa_passed_pending_user_acceptance",
        },
    )

    preflight = runner.preflight_batch(
        PRODUCTION_JOBS,
        tmp_path / "candidates",
        authenticate_model=False,
        require_base_qa=True,
    )

    assert preflight["execution_authorized"] is False
    with pytest.raises(ValueError, match="execution-authorized"):
        runner.run_preflighted_jobs(preflight, tmp_path / "candidates", object())


def test_configuration_only_preflight_cannot_load_or_execute(tmp_path):
    preflight = {
        "schema": "flux2_human_attribute_preflight_v2",
        "gpu_touched": False,
        "execution_authorized": False,
        "runner": {"sha256": runner.sha256_file(runner.RUNNER_PATH)},
        "model": {"revision": runner.MODEL_REVISION, "snapshot": str(tmp_path)},
        "output_root": str(tmp_path),
        "jobs": [],
    }

    with pytest.raises(ValueError, match="execution-authorized"):
        runner.load_pipeline(preflight)
    with pytest.raises(ValueError, match="execution-authorized"):
        runner.run_preflighted_jobs(preflight, tmp_path, object())


def test_model_snapshot_authenticates_exact_inventory_license_and_rejects_incomplete(tmp_path):
    root = tmp_path / "model"
    snapshot = root / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (snapshot / "LICENSE.md").write_text("license", encoding="utf-8")
    records = []
    for path in sorted(snapshot.iterdir()):
        records.append(
            {
                "relative_path": path.name,
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "huggingface_snapshot_inventory_v1",
                "model": "test/model",
                "revision": "revision",
                "files": records,
                "license_relative_path": "LICENSE.md",
            }
        ),
        encoding="utf-8",
    )

    result = runner.authenticate_model_snapshot(
        model_root=root,
        revision="revision",
        inventory_path=inventory,
        inventory_sha256=_sha(inventory),
        expected_model_name="test/model",
    )

    assert result["file_count"] == 2
    assert result["license"]["sha256"] == _sha(snapshot / "LICENSE.md")
    incomplete = root / "blobs" / "bad.incomplete"
    incomplete.parent.mkdir()
    incomplete.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        runner.authenticate_model_snapshot(
            model_root=root,
            revision="revision",
            inventory_path=inventory,
            inventory_sha256=_sha(inventory),
            expected_model_name="test/model",
        )


def test_main_completes_all_preflight_before_pipeline_load(monkeypatch, tmp_path):
    calls = []

    def fake_preflight(jobs, output, authenticate_model=True):
        calls.append("preflight")
        return {"jobs": [], "jobs_sha256": "a" * 64}

    def fake_load(preflight):
        calls.append("load_pipeline")
        return object()

    def fake_run(preflight, output, pipeline):
        calls.append("run_jobs")
        return []

    monkeypatch.setattr(runner, "preflight_batch", fake_preflight)
    monkeypatch.setattr(runner, "load_pipeline", fake_load)
    monkeypatch.setattr(runner, "run_preflighted_jobs", fake_run)

    result = runner.main(
        [
            "--jobs-json",
            str(PRODUCTION_JOBS),
            "--output-root",
            str(tmp_path),
            "--local-files-only",
        ]
    )

    assert result == 0
    assert calls == ["preflight", "load_pipeline", "run_jobs"]


def test_main_treats_authenticated_resume_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "preflight_batch", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runner, "load_pipeline", lambda preflight: object())
    monkeypatch.setattr(
        runner,
        "run_preflighted_jobs",
        lambda preflight, output, pipeline: [
            {"case_id": "hat", "status": "existing_success"}
        ],
    )

    assert (
        runner.main(
            [
                "--jobs-json",
                str(PRODUCTION_JOBS),
                "--output-root",
                str(tmp_path),
                "--local-files-only",
            ]
        )
        == 0
    )


def test_attempt_failure_preserves_unique_immutable_ledger_and_staging(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    def operation(staging: Path):
        (staging / "raw_candidate.png").write_bytes(b"raw evidence")
        raise RuntimeError("synthetic failure")

    first = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )
    second = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )

    assert first["status"] == second["status"] == "rejected_generation_failure"
    assert first["attempt_id"] != second["attempt_id"]
    for result in (first, second):
        ledger = Path(result["ledger"])
        evidence = Path(result["evidence_dir"])
        assert ledger.stat().st_mode & 0o777 == 0o444
        assert (evidence / "raw_candidate.png").read_bytes() == b"raw evidence"
        assert all(path.stat().st_mode & 0o222 == 0 for path in evidence.rglob("*") if path.is_file())
        payload = json.loads(ledger.read_text())
        assert payload["error"] == {"type": "RuntimeError", "message": "synthetic failure"}


def test_attempt_start_ledger_is_durable_before_operation(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    observed = {"durable": False}

    def operation(staging: Path):
        ledgers = list((output_root / ".attempts/hat").glob("*.json"))
        assert len(ledgers) == 1
        ledger = ledgers[0]
        payload = json.loads(ledger.read_text())
        assert payload["status"] == "started"
        assert payload["staging"]["path"] == str(staging)
        assert ledger.stat().st_mode & 0o777 == 0o444
        observed["durable"] = True
        raise RuntimeError("stop after durable start")

    result = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )

    assert result["status"] == "rejected_generation_failure"
    assert observed["durable"] is True


def test_success_ledger_is_sealed_in_staging_before_atomic_publication(
    monkeypatch, tmp_path
):
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    def operation(staging: Path):
        (staging / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "schema": runner.OUTPUT_SCHEMA,
                    "case_id": "hat",
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
        return "complete"

    real_rename = runner._rename_noreplace

    def inspected_rename(source: Path, destination: Path):
        if destination == output_root / "hat":
            ledger = source / "generation_attempt.json"
            assert ledger.is_file()
            assert ledger.stat().st_mode & 0o777 == 0o444
            assert json.loads(ledger.read_text())["status"] == "generated"
            manifest = json.loads((source / "candidate_manifest.json").read_text())
            assert manifest["artifacts"]["generation_attempt.json"]["sha256"] == _sha(
                ledger
            )
        real_rename(source, destination)

    monkeypatch.setattr(runner, "_rename_noreplace", inspected_rename)
    result = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )

    assert result["status"] == "generated"
    assert (output_root / "hat/generation_attempt.json").is_file()


def test_authenticated_inference_is_bracketed_by_twice_stable_reauthentication(
    monkeypatch
):
    events = []
    snapshots = iter(({"generation": 1}, {"generation": 1}))
    record = {"job": {"case_id": "short_sleeve_color"}}
    image = Image.new("RGB", (2, 2), "white")

    monkeypatch.setattr(
        runner,
        "reauthenticate_execution_inputs",
        lambda preflight, supplied: events.append("reauth") or next(snapshots),
    )
    monkeypatch.setattr(
        runner,
        "_run_flux_inference",
        lambda supplied, pipeline: events.append("inference") or image,
    )

    raw, alpha, proof = runner.run_authenticated_inference(
        {"execution_authorized": True}, record, object(), alpha_predictor=lambda *_: None
    )

    assert raw is image
    assert alpha is None
    assert proof == {
        "preflight_reauthenticated": True,
        "postflight_reauthenticated": True,
        "preflight_snapshot": {"generation": 1},
        "postflight_snapshot": {"generation": 1},
    }
    assert events == ["reauth", "inference", "reauth"]


def test_execution_reauthentication_requires_two_identical_current_snapshots(
    monkeypatch,
):
    calls = []
    expected = {"snapshot": "pinned"}
    monkeypatch.setattr(runner, "_assert_execution_preflight", lambda value: None)
    monkeypatch.setattr(
        runner,
        "_execution_snapshot_expected",
        lambda preflight, record: expected,
    )
    monkeypatch.setattr(
        runner,
        "_current_execution_input_snapshot",
        lambda preflight, record: calls.append("read") or expected,
    )

    assert runner.reauthenticate_execution_inputs({}, {"job": {}}) == expected
    assert calls == ["read", "read"]


def test_execution_reauthentication_rejects_stable_but_stale_snapshot(monkeypatch):
    monkeypatch.setattr(runner, "_assert_execution_preflight", lambda value: None)
    monkeypatch.setattr(
        runner,
        "_execution_snapshot_expected",
        lambda preflight, record: {"snapshot": "pinned"},
    )
    monkeypatch.setattr(
        runner,
        "_current_execution_input_snapshot",
        lambda preflight, record: {"snapshot": "replaced"},
    )

    with pytest.raises(ValueError, match="changed after execution preflight"):
        runner.reauthenticate_execution_inputs({}, {"job": {}})


def test_attempt_does_not_swallow_keyboard_interrupt(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    with pytest.raises(KeyboardInterrupt):
        runner.execute_with_attempt(
            case_id="hat",
            output_root=output_root,
            job_descriptor={"sha256": "a" * 64},
            operation=lambda staging: (_ for _ in ()).throw(KeyboardInterrupt()),
        )


def test_attempt_missing_manifest_is_failure_evidence_not_published_destination(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    def operation(staging: Path):
        (staging / "raw_candidate.png").write_bytes(b"incomplete")

    result = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )

    assert result["status"] == "rejected_generation_failure"
    assert not (output_root / "hat").exists()
    evidence = Path(result["evidence_dir"])
    assert (evidence / "raw_candidate.png").read_bytes() == b"incomplete"


def test_incomplete_existing_success_is_rejected_as_unauthenticated(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    destination = output_root / "hat"
    destination.mkdir()
    manifest = destination / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": runner.OUTPUT_SCHEMA,
                "case_id": "hat",
                "state_classification": "research_candidate",
                "bundle_status": "generated_pending_agent_2d_visual_qa",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authenticated candidate snapshot"):
        runner.execute_with_attempt(
            case_id="hat",
            output_root=output_root,
            job_descriptor={"sha256": "a" * 64},
            operation=lambda staging: None,
        )


def test_existing_full_authenticated_success_is_resumed_without_rejection(
    monkeypatch, tmp_path
):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    destination = output_root / "hat"
    destination.mkdir()
    manifest = destination / "candidate_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner.attribute_review,
        "validated_candidate_snapshot",
        lambda bundle: {
            "case_id": "hat",
            "candidate_manifest_path": str(manifest),
        },
    )
    called = False

    def operation(staging: Path):
        nonlocal called
        called = True

    result = runner.execute_with_attempt(
        case_id="hat",
        output_root=output_root,
        job_descriptor={"sha256": "a" * 64},
        operation=operation,
    )

    assert result["status"] == "existing_success"
    assert called is False
    assert not list(output_root.glob("hat.rejected*"))


def test_success_bundle_contains_rgba_masks_overlay_diff_and_pending_immutable_agent_decision(
    tmp_path, monkeypatch,
):
    preflight = runner.preflight_batch(
        PRODUCTION_JOBS,
        tmp_path / "candidates",
        authenticate_model=False,
        require_base_qa=False,
    )
    record = next(
        item for item in preflight["jobs"] if item["job"]["case_id"] == "short_sleeve_color"
    )
    record = copy.deepcopy(record)
    record["base_qualification"] = {
        "asset_id": record["job"]["base_asset_id"],
        "review_dir": "/qualified/dynamic_review_v1",
        "status": "agent_qa_passed_pending_user_acceptance",
        "qualified_candidate": {
            "path": record["job"]["base_qualified_candidate"],
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "final_branch": {
            "branch_id": "direct",
            "path": "/qualified",
            "relative_root": ".",
        },
    }
    with Image.open(record["source"]["image"]) as opened:
        raw = Image.new("RGB", opened.size, (20, 70, 210))
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        runner.semantic_masks,
        "evaluate_candidate_metrics",
        lambda *args, **kwargs: {
            "case_id": "short_sleeve_color",
            "passed": True,
            "checks": {"fixture_quantitative_gate": True},
            "metrics": {"fixture_only": True},
        },
    )

    manifest = runner.build_candidate_bundle(
        staging=staging,
        public_destination=staging,
        preflight_record=record,
        jobs_descriptor=preflight["jobs_descriptor"],
        raw_candidate=raw,
        predicted_alpha=None,
    )

    expected = {
        "source.png",
        "raw_candidate.png",
        "candidate.png",
        "source_alpha.png",
        "candidate_alpha.png",
        "candidate_rgba.png",
        "edit_core.png",
        "transition_band.png",
        "protected_guard.png",
        "overlay.png",
        "diff.png",
        "agent_2d_decision.json",
        "candidate_manifest.json",
    }
    assert {path.name for path in staging.iterdir()} == expected
    payload = json.loads(manifest.read_text())
    assert payload["schema"] == runner.OUTPUT_SCHEMA
    assert payload["bundle_status"] == "generated_pending_agent_2d_visual_qa"
    assert payload["base_route2_qualification"] == record["base_qualification"]
    assert set(payload["artifacts"]) == expected - {"candidate_manifest.json"}
    decision = json.loads((staging / "agent_2d_decision.json").read_text())
    assert decision["status"] == "pending_agent_2d_visual_qa"
    assert decision["user_acceptance"] == "pending_user_review"
    assert "user_approved" not in json.dumps(decision)
    assert (staging / "agent_2d_decision.json").stat().st_mode & 0o222 == 0
    with Image.open(staging / "source_alpha.png") as source_alpha, Image.open(
        staging / "candidate_alpha.png"
    ) as candidate_alpha:
        assert source_alpha.tobytes() == candidate_alpha.tobytes()
    with Image.open(staging / "candidate_rgba.png") as rgba:
        assert rgba.mode == "RGBA"
        assert rgba.getchannel("A").getextrema() == (0, 255)
    with Image.open(staging / "source.png") as source, Image.open(
        staging / "candidate.png"
    ) as candidate, Image.open(staging / "edit_core.png") as core, Image.open(
        staging / "transition_band.png"
    ) as band:
        outside = ~((np.asarray(core) > 0) | (np.asarray(band) > 0))
        assert np.array_equal(np.asarray(source)[outside], np.asarray(candidate)[outside])


def test_candidate_builder_rejects_configuration_only_base_record(tmp_path):
    preflight = runner.preflight_batch(
        PRODUCTION_JOBS,
        tmp_path / "candidates",
        authenticate_model=False,
        require_base_qa=False,
    )
    record = preflight["jobs"][0]
    with Image.open(record["source"]["image"]) as opened:
        raw = Image.new("RGB", opened.size, (20, 70, 210))
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="qualified Route-2 base"):
        runner.build_candidate_bundle(
            staging=staging,
            public_destination=staging,
            preflight_record=record,
            jobs_descriptor=preflight["jobs_descriptor"],
            raw_candidate=raw,
            predicted_alpha=Image.new("L", raw.size, 255),
        )


def test_candidate_builder_rejects_changed_pixels_when_tall_height_gate_fails(
    tmp_path, monkeypatch
):
    production_runner_hash = json.loads(PRODUCTION_JOBS.read_text())["jobs"][0][
        "runner_sha256"
    ]
    real_sha256_file = runner.sha256_file

    def legacy_jobs_hash(path):
        return (
            production_runner_hash
            if Path(path).absolute() == runner.RUNNER_PATH.absolute()
            else real_sha256_file(path)
        )

    monkeypatch.setattr(runner, "sha256_file", legacy_jobs_hash)
    preflight = runner.preflight_batch(
        PRODUCTION_JOBS,
        tmp_path / "candidates",
        authenticate_model=False,
        require_base_qa=False,
    )
    record = copy.deepcopy(preflight["jobs"][0])
    record["base_qualification"] = {
        "asset_id": record["job"]["base_asset_id"],
        "review_dir": "/qualified/dynamic_review_v1",
        "status": "agent_qa_passed_pending_user_acceptance",
        "qualified_candidate": {
            "path": record["job"]["base_qualified_candidate"],
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "final_branch": {
            "branch_id": "sanitized_weights",
            "path": "/qualified/fitted_skeleton_v1/sanitized_weights_v1",
            "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
        },
    }
    with Image.open(record["source"]["image"]) as opened:
        raw = Image.new("RGB", opened.size, (40, 90, 140))
    with Image.open(record["source"]["source_alpha"]["path"]) as opened:
        unchanged_alpha = opened.convert("L")
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="case-specific quantitative gates"):
        runner.build_candidate_bundle(
            staging=staging,
            public_destination=tmp_path / "tall_man",
            preflight_record=record,
            jobs_descriptor=preflight["jobs_descriptor"],
            raw_candidate=raw,
            predicted_alpha=unchanged_alpha,
        )
