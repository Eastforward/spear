import ast
import copy
from pathlib import Path

import pytest

from tools import audit_controlled_animal_one_shot_batch as auditor
from tools import controlled_source_asset_schema as contracts


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/audit_controlled_animal_one_shot_batch.py"
)


def test_one_shot_auditor_is_parseable_and_has_no_seed_override_cli():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)

    assert "--flux-batch" in source
    assert "--pixal-inputs" in source
    assert "--pixal-batch" in source
    assert "--output" in source
    assert "--seed" not in source
    assert "best_of" not in source


def _candidate(*, ordinal):
    return {
        "index": {
            "candidate_manifest": {"sha256": f"{ordinal:x}" * 64},
        },
        "manifest": {
            "execution_job_id": f"animal_{ordinal}",
            "request_sha256": f"{ordinal + 2:x}" * 64,
            "profile_schema_id": f"dog_profile_{ordinal}",
            "generation": {
                "seed": 40 + ordinal,
                "flux_invocations": 1,
            },
            "output": {"sha256": f"{ordinal + 4:x}" * 64},
        },
    }


def _bounded_fixture(tmp_path, monkeypatch):
    flux_path = tmp_path / "flux_batch.json"
    flux_path.write_text("{}\n", encoding="utf-8")
    batch = {
        "batch_sha256": "a" * 64,
        "selection": {"bounded_exploration": {"policy": {}, "groups": []}},
    }
    candidates = {
        "dog_rejected": _candidate(ordinal=1),
        "dog_selected": _candidate(ordinal=2),
    }
    monkeypatch.setattr(
        auditor.review,
        "load_flux_batch",
        lambda path: (tmp_path, batch, candidates),
    )
    return flux_path, candidates


def test_bounded_audit_fails_closed_without_authenticated_pixal_inputs(
    tmp_path, monkeypatch
):
    flux_path, _candidates = _bounded_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        contracts.ContractError,
        match="requires --pixal-inputs",
    ):
        auditor.build_audit(flux_path)


def test_ordinary_one_shot_output_shape_is_unchanged_without_pixal_inputs(
    tmp_path, monkeypatch
):
    flux_path = tmp_path / "flux_batch.json"
    flux_path.write_text("{}\n", encoding="utf-8")
    batch = {"batch_sha256": "a" * 64, "selection": {}}
    candidates = {"dog_selected": _candidate(ordinal=2)}
    evidence = {
        "mode": "native_policy_enforced_before_inference",
        "profile_qualification_authorized": True,
    }
    monkeypatch.setattr(
        auditor.review,
        "load_flux_batch",
        lambda path: (tmp_path, batch, candidates),
    )
    monkeypatch.setattr(
        auditor.pixal_inputs,
        "_flux_one_shot_evidence",
        lambda loaded_batch, loaded_candidates: copy.deepcopy(evidence),
    )
    monkeypatch.setattr(
        auditor.pixal_runner,
        "load_pixal_inputs",
        lambda path: pytest.fail("ordinary audit unexpectedly loaded Pixal inputs"),
    )

    result = auditor.build_audit(flux_path)

    assert set(result) == {
        "schema",
        "status",
        "flux_batch",
        "one_shot_evidence",
        "candidate_count",
        "candidates",
        "automatic_checks",
        "formal_dataset_registration_authorized",
        "audit_sha256",
    }
    assert set(result["automatic_checks"]) == {
        "batch_and_candidate_hashes_reauthenticated",
        "one_recorded_candidate_per_request",
        "one_recorded_flux_invocation_per_candidate",
        "profile_qualification_authorized",
        "overall",
    }
    assert result["status"] == "passed_native_policy"


def test_bounded_pixal_coverage_uses_only_frozen_selected_jobs(
    tmp_path, monkeypatch
):
    flux_path, candidates = _bounded_fixture(tmp_path, monkeypatch)
    pixal_inputs_path = tmp_path / "pixal_inputs_manifest.json"
    pixal_inputs_path.write_text("{}\n", encoding="utf-8")
    selected = candidates["dog_selected"]["manifest"]
    freeze = {"authenticated": True}
    evidence = {
        "mode": "native_bounded_exploration_frozen",
        "profile_qualification_authorized": True,
    }
    pixal_payload = {
        "manifest_sha256": "b" * 64,
        "upstream_flux_one_shot_evidence": copy.deepcopy(evidence),
        "bounded_exploration_freeze": freeze,
        "jobs": [
            {
                "controlled_request": {
                    "instance_id": "dog_selected",
                    "execution_job_id": selected["execution_job_id"],
                    "request_sha256": selected["request_sha256"],
                    "profile_schema_id": selected["profile_schema_id"],
                    "generation_seed": selected["generation"]["seed"],
                },
                "reference": {
                    "source": {"sha256": selected["output"]["sha256"]}
                },
            }
        ],
    }
    loaded_paths = []

    def load_pixal_inputs(path):
        loaded_paths.append(Path(path).resolve())
        return Path(path).resolve(), copy.deepcopy(pixal_payload)

    def flux_evidence(batch, loaded_candidates, bounded_exploration_freeze=None):
        assert batch["batch_sha256"] == "a" * 64
        assert loaded_candidates is candidates
        assert bounded_exploration_freeze == freeze
        return copy.deepcopy(evidence)

    covered = []

    def audit_pixal_batch(path, rows):
        covered.extend(row["instance_id"] for row in rows)
        return {"path": str(path)}

    monkeypatch.setattr(
        auditor.pixal_runner, "load_pixal_inputs", load_pixal_inputs
    )
    monkeypatch.setattr(
        auditor.pixal_inputs, "_flux_one_shot_evidence", flux_evidence
    )
    monkeypatch.setattr(auditor, "_audit_pixal_batch", audit_pixal_batch)

    result = auditor.build_audit(
        flux_path,
        pixal_batch_path=tmp_path / "pixal_batch.json",
        pixal_inputs_path=pixal_inputs_path,
    )

    assert loaded_paths == [pixal_inputs_path.resolve()]
    assert covered == ["dog_selected"]
    assert result["candidate_count"] == 2
    assert result["status"] == "passed_native_policy"
    assert result["automatic_checks"]["profile_qualification_authorized"] is True
    assert (
        result["automatic_checks"][
            "bounded_exploration_freeze_reauthenticated_by_pixal_input_loader"
        ]
        is True
    )
