from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import flux2_edit_human_attributes as runner


SPEAR_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_JOBS = SPEAR_ROOT / "tmp/human_attribute_instances_v1/jobs_v2.json"


def _payload() -> dict:
    return json.loads(PRODUCTION_JOBS.read_text(encoding="utf-8"))


def _write_jobs(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exact_order_gender_sources_and_model_revision_are_frozen():
    assert tuple(runner.CASE_CONTRACTS) == (
        "tall_man",
        "short_woman",
        "glasses",
        "hat",
        "short_sleeve_color",
        "trousers",
        "shoes",
    )
    assert runner.CASE_CONTRACTS["tall_man"]["base_asset_id"] == (
        "rocketbox_male_adult_01"
    )
    assert runner.CASE_CONTRACTS["short_woman"]["base_asset_id"] == (
        "rocketbox_female_adult_01"
    )
    assert runner.MODEL_REVISION == "e7b7dc27f91deacad38e78976d1f2b499d76a294"


def test_job_contract_rejects_wrong_order_and_mutable_generation_values(tmp_path):
    payload = _payload()
    path = tmp_path / "jobs.json"
    _write_jobs(path, payload)
    assert [item["case_id"] for item in runner.load_jobs(path)] == list(
        runner.CASE_CONTRACTS
    )

    wrong_order = copy.deepcopy(payload)
    wrong_order["jobs"].reverse()
    _write_jobs(path, wrong_order)
    with pytest.raises(ValueError, match="exact case order"):
        runner.load_jobs(path)

    wrong_canvas = copy.deepcopy(payload)
    wrong_canvas["jobs"][0]["width"] = 1024
    _write_jobs(path, wrong_canvas)
    with pytest.raises(ValueError, match="1152x1536"):
        runner.load_jobs(path)

    online_model = copy.deepcopy(payload)
    online_model["jobs"][0]["model"]["local_files_only"] = False
    _write_jobs(path, online_model)
    with pytest.raises(ValueError, match="model/local-only/inventory"):
        runner.load_jobs(path)


def test_authenticate_source_requires_recursive_approved_snapshot():
    job = runner.load_jobs(PRODUCTION_JOBS)[0]

    result = runner.authenticate_source(job)

    assert result["image_sha256"] == job["source_image_sha256"]
    assert result["review_sha256"] == job["source_review_sha256"]
    assert result["candidate_manifest"]["sha256"] == job[
        "source_candidate_manifest"
    ]["sha256"]
    assert result["source_alpha"]["sha256"] == job["source_alpha"]["sha256"]
    assert result["source_rgba"]["sha256"] == job["source_rgba"]["sha256"]
    assert result["source_rgba_rgb_matches_source"] is True

    tampered = copy.deepcopy(job)
    tampered["source_image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        runner.authenticate_source(tampered)


def test_removed_v1_mask_and_rejection_shortcuts_do_not_return():
    for name in (
        "build_allowed_mask",
        "masked_composite",
        "write_rejection_evidence",
        "run_one",
        "run_jobs",
    ):
        assert not hasattr(runner, name)
