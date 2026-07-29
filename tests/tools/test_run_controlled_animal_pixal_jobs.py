from pathlib import Path
import copy

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import controlled_animal_one_shot_policy as one_shot
from tools import pixal_animal_persistent_worker as persistent_worker
from tools import review_controlled_animal_flux2_candidates as flux_review
from tools import run_controlled_animal_pixal_jobs as runner


def test_persistent_worker_defaults_to_full_gpu_residency():
    args = persistent_worker.parse_args(
        ["--jobs", "jobs.json", "--gpu", "3", "--status", "status.json"]
    )

    assert args.low_vram is False
    assert args.standard_vram is False


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


def test_build_worker_orders_exposes_one_shared_rotated_queue_per_gpu():
    jobs = [{"legacy_tag": f"animal_{index}"} for index in range(10)]

    orders = runner.build_worker_orders(jobs, [0, 1, 2, 3])

    expected = {job["legacy_tag"] for job in jobs}
    assert set(orders) == {0, 1, 2, 3}
    assert all(len(order) == len(jobs) for order in orders.values())
    assert all(
        {job["legacy_tag"] for job in order} == expected
        for order in orders.values()
    )
    assert [orders[gpu][0]["legacy_tag"] for gpu in [0, 1, 2, 3]] == [
        "animal_0",
        "animal_1",
        "animal_2",
        "animal_3",
    ]


def test_claim_job_is_atomic_and_records_owner(tmp_path):
    job = {"legacy_tag": "dog_pug_test"}

    first = persistent_worker.claim_job(tmp_path, job, gpu=2)
    second = persistent_worker.claim_job(tmp_path, job, gpu=3)

    assert first is not None
    assert second is None
    payload = contracts.load_json(first)
    assert payload["schema"] == "pixal_dynamic_work_claim_v1"
    assert payload["legacy_tag"] == "dog_pug_test"
    assert payload["gpu"] == 2
    assert payload["claim_sha256"] == persistent_worker.claim_id("dog_pug_test")


def test_build_worker_job_separates_staging_write_from_public_path(tmp_path):
    public_root = tmp_path / "published"
    staging = tmp_path / ".published.staging"
    job = {
        "legacy_tag": "dog_x",
        "candidate_tag": "dog_x_pixal_v1",
        "seed": 42,
        "attempt_ordinal": 0,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "reference": {"pixal_input": {"path": "/input.png"}},
        "controlled_request": {
            "instance_id": "dog_x",
            "execution_job_id": "animal_x",
            "request_sha256": "0" * 64,
            "profile_schema_id": "dog_x_v1",
            "sampled_attributes": {"size": "small"},
            "target_physical_profile": {},
            "generation_seed": 42,
        },
    }

    worker_job = runner.build_worker_job(job, staging, public_root)

    assert Path(worker_job["output"]) == staging / "dog_x/pixal_raw_1024.glb"
    assert Path(worker_job["manifest"]) == staging / "dog_x/pixal_raw_1024.manifest.json"
    assert Path(worker_job["public_output"]) == public_root / "dog_x/pixal_raw_1024.glb"
    assert Path(worker_job["public_manifest"]) == public_root / "dog_x/pixal_raw_1024.manifest.json"
    assert worker_job["controlled_request"] == job["controlled_request"]
    assert worker_job["attempt_ordinal"] == 0
    assert worker_job["one_shot_execution"] == one_shot.stage_record("pixal3d")


def test_build_worker_job_reject_contract_detects_seed_change():
    job = {
        "seed": 43,
        "attempt_ordinal": 0,
        "controlled_request": {"generation_seed": 42},
        "one_shot_execution": one_shot.stage_record("pixal3d"),
    }

    with pytest.raises(one_shot.PolicyError, match="frozen request"):
        one_shot.validate_pixal_job(job)


def test_pixal_loader_binding_rechecks_bounded_exploration_receipt_hashes():
    profile_id = "dog_short_leg_v1"
    jobs = [
        {
            "execution_job_id": f"animal_{index}",
            "profile_schema_id": profile_id,
            "profile_sha256": "1" * 64,
            "consumer_requests": [
                {
                    "instance_id": f"dog_{index}",
                    "request_sha256": f"{index + 2:x}" * 64,
                }
            ],
        }
        for index in range(2)
    ]
    group = one_shot.build_bounded_exploration_group(
        profile_schema_id=profile_id,
        profile_sha256="1" * 64,
        execution_preflight_sha256="4" * 64,
        request_batch_sha256="5" * 64,
        jobs=jobs,
    )
    reviews = [
        {
            "instance_id": f"dog_{index}",
            "profile_schema_id": profile_id,
            "candidate_sha256": f"{index + 6:x}" * 64,
            "decision": "approved_for_pixal3d" if index == 1 else "rejected",
            "review": {"sha256": f"{index + 8:x}" * 64},
        }
        for index in range(2)
    ]
    receipt = one_shot.build_bounded_exploration_freeze(
        group=group,
        candidate_reviews=reviews,
        flux_batch_sha256="a" * 64,
    )
    policy = one_shot.bounded_exploration_policy_record()
    evidence = {
        "mode": "native_bounded_exploration_frozen",
        "bounded_exploration_policy": policy,
        "freeze_receipt_sha256s": [receipt["freeze_receipt_sha256"]],
    }
    payload = {
        "bounded_exploration_freeze": {
            "policy": policy,
            "review_batch_sha256": "b" * 64,
            "freeze_receipts": [receipt],
        }
    }
    pixal_jobs = [
        {
            "controlled_request": {"instance_id": "dog_1"},
            "reference": {"source": {"sha256": "7" * 64}},
        }
    ]

    runner._validate_bounded_exploration_binding(payload, evidence, pixal_jobs)

    with pytest.raises(contracts.ContractError, match="missing"):
        runner._validate_bounded_exploration_binding({}, evidence, pixal_jobs)
    pixal_jobs[0]["reference"]["source"]["sha256"] = "f" * 64
    with pytest.raises(contracts.ContractError, match="differ"):
        runner._validate_bounded_exploration_binding(
            payload, evidence, pixal_jobs
        )


def test_pixal_loader_reopens_bounded_review_source_before_accepting_freeze(
    tmp_path, monkeypatch
):
    review_path = tmp_path / "review_batch.json"
    flux_path = tmp_path / "flux_batch.json"
    review_path.write_text("{}\n", encoding="utf-8")
    flux_batch = {
        "schema": flux_review.flux_runner.BATCH_SCHEMA,
        "status": "pending_2d_review",
        "selection": {"bounded_exploration": {"declared": True}},
    }
    flux_batch["batch_sha256"] = runner._hash_without(
        flux_batch, "batch_sha256"
    )
    contracts.write_json_no_replace(flux_path, flux_batch)
    expected_freeze = {
        "policy": {"bounded": True},
        "review_batch_sha256": "b" * 64,
        "freeze_receipts": [{"freeze": "exact"}],
    }
    review_batch = {
        "review_batch_sha256": "b" * 64,
        "flux2_batch": {
            "path": str(flux_path),
            "sha256": runner._sha256_file(flux_path),
            "batch_sha256": flux_batch["batch_sha256"],
        },
    }
    payload = {
        "review_batch": {
            "path": str(review_path),
            "sha256": runner._sha256_file(review_path),
            "review_batch_sha256": "b" * 64,
        },
        "bounded_exploration_freeze": copy.deepcopy(expected_freeze),
    }
    evidence = {
        "mode": "native_bounded_exploration_frozen",
        "flux_batch_sha256": flux_batch["batch_sha256"],
    }
    candidates = {"selected": {}}

    monkeypatch.setattr(
        runner.pixal_inputs,
        "load_review_batch",
        lambda _path: copy.deepcopy(review_batch),
    )
    monkeypatch.setattr(
        flux_review,
        "load_flux_batch",
        lambda _path: (tmp_path, copy.deepcopy(flux_batch), candidates),
    )
    def load_review_payloads(*_args, **kwargs):
        assert kwargs["bounded_exploration"] is True
        return {
            "selected": {
                "decision": "approved_for_pixal3d",
                "candidate": {"sha256": "c" * 64},
            }
        }

    monkeypatch.setattr(
        runner.pixal_inputs,
        "_load_authenticated_review_payloads",
        load_review_payloads,
    )
    monkeypatch.setattr(
        runner.pixal_inputs,
        "_load_authenticated_preflight",
        lambda _batch: {"preflight_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        runner.pixal_inputs,
        "_authenticated_route_jobs",
        lambda *_args: ("flux2_pixal3d_animal_v1", {}, {}),
    )
    monkeypatch.setattr(
        runner.pixal_inputs,
        "_validate_bounded_exploration_declaration",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runner.pixal_inputs,
        "_validate_bounded_exploration_freezes",
        lambda **_kwargs: copy.deepcopy(expected_freeze),
    )

    runner._reauthenticate_bounded_exploration_source(payload, evidence)

    payload["bounded_exploration_freeze"]["freeze_receipts"] = [
        {"freeze": "replacement"}
    ]
    with pytest.raises(contracts.ContractError, match="differs from its review source"):
        runner._reauthenticate_bounded_exploration_source(payload, evidence)


def test_pixal_loader_rejects_self_hashed_bounded_to_ordinary_downgrade(
    tmp_path,
):
    flux_path = tmp_path / "flux_batch.json"
    flux_batch = {
        "schema": flux_review.flux_runner.BATCH_SCHEMA,
        "status": "pending_2d_review",
        "selection": {"bounded_exploration": {"declared": True}},
    }
    flux_batch["batch_sha256"] = runner._hash_without(
        flux_batch, "batch_sha256"
    )
    contracts.write_json_no_replace(flux_path, flux_batch)

    review_path = tmp_path / "review_batch.json"
    review_batch = {
        "schema": flux_review.BATCH_REVIEW_SCHEMA,
        "flux2_batch": {
            "path": str(flux_path),
            "sha256": runner._sha256_file(flux_path),
            "batch_sha256": flux_batch["batch_sha256"],
        },
        "automatic_checks": {"overall": "passed"},
    }
    review_batch["review_batch_sha256"] = runner._hash_without(
        review_batch, "review_batch_sha256"
    )
    contracts.write_json_no_replace(review_path, review_batch)

    manifest_path = tmp_path / "pixal_inputs_manifest.json"
    payload = {
        "schema": runner.pixal_inputs.PIXAL_INPUT_SCHEMA,
        "status": "ready_for_pixal3d",
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "upstream_flux_one_shot_evidence": {
            "mode": "native_policy_enforced_before_inference",
            "policy": one_shot.policy_record(),
            "flux_batch_sha256": flux_batch["batch_sha256"],
            "profile_qualification_authorized": True,
        },
        "review_batch": {
            "path": str(review_path),
            "sha256": runner._sha256_file(review_path),
            "review_batch_sha256": review_batch["review_batch_sha256"],
        },
        "job_count": 1,
        "jobs": [{"controlled_request": {"instance_id": "unfrozen_candidate"}}],
        "automatic_checks": {"overall": "passed"},
    }
    payload["manifest_sha256"] = runner._hash_without(
        payload, "manifest_sha256"
    )
    contracts.write_json_no_replace(manifest_path, payload)

    with pytest.raises(
        contracts.ContractError,
        match="cannot be downgraded",
    ):
        runner.load_pixal_inputs(manifest_path)
