import copy
import hashlib
from pathlib import Path

import pytest

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_source_asset_execution as preparation
from tools import register_controlled_animal_source_assets as registry


PROFILE = (
    Path(__file__).resolve().parents[2]
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "dog_shiba_inu_four_limb_rest_side_clay_v1.json"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_without(value, key):
    return registry._hash_without(value, key)


def _relative_record(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _absolute_record(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def frozen_historical_preflight(tmp_path):
    profiles = input_builder.load_profiles([PROFILE])
    roots = preparation.default_artifact_roots()
    authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, roots
        )
        for profile in profiles
    }
    files = input_builder.compile_inputs(
        profiles=profiles,
        count_per_profile=1,
        seed=20260728,
        plan_id="frozen_historical_preflight_test_v1",
        split_salt="frozen_historical_preflight_test_v1",
        max_qa_pairs_per_split=None,
        artifact_authentication=authentication,
    )
    historical_routes = {
        "flux2_pixal3d_animal_v1",
        "stable_animal_template_v1",
        "rocketbox_material_v1",
    }
    files["execution_jobs.json"] = input_builder.build_execution_jobs(
        files["instance_requests.json"], route_names=historical_routes
    )
    input_dir = tmp_path / "inputs"
    input_builder.publish_output(input_dir, files)
    current = preparation.build_execution_preflight(input_dir, roots)
    historical = copy.deepcopy(current)
    historical["routes"].pop("flux2_pixal3d_static_v1")
    historical["execution_summary"].pop("static_object_job_count")
    historical["preflight_sha256"] = preparation.preflight_sha256(historical)
    path = tmp_path / "execution_preflight.json"
    contracts.write_json_no_replace(path, historical)
    return path, input_dir


def test_spear_artifact_rejects_paths_outside_repo(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"fixture")

    with pytest.raises(contracts.ContractError, match="outside SPEAR"):
        registry.spear_artifact(path)


def test_spear_artifact_rejects_leaf_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "SPEAR_ROOT", tmp_path)
    direct = tmp_path / "direct.bin"
    direct.write_bytes(b"fixture")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(direct)

    with pytest.raises(contracts.ContractError, match="missing/non-direct"):
        registry.spear_artifact(alias)


def test_pinned_model_license_snapshots_are_present_and_hashed():
    records = registry.license_records()

    assert len(records) == 4
    assert {record["root_id"] for record in records} == {"models_root"}
    assert all(len(record["sha256"]) == 64 for record in records)
    for record in records:
        direct = registry.MODELS_ROOT / record["path"]
        assert direct.absolute() == direct.resolve(strict=True)
        assert direct.stat().st_size == record["size_bytes"]
        assert _sha256(direct) == record["sha256"]


def test_license_records_replace_snapshot_symlink_with_direct_blob(
    tmp_path, monkeypatch
):
    blob = tmp_path / "blobs" / "license-bytes"
    blob.parent.mkdir()
    blob.write_bytes(b"pinned license")
    snapshot = tmp_path / "snapshots" / "revision" / "LICENSE"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    monkeypatch.setattr(registry, "MODELS_ROOT", tmp_path)
    monkeypatch.setattr(
        registry,
        "LICENSE_SPECS",
        (
            {
                "path": snapshot.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(blob),
                "size_bytes": blob.stat().st_size,
            },
        ),
    )

    records = registry.license_records()

    assert records == [
        {
            "root_id": "models_root",
            "path": "blobs/license-bytes",
            "sha256": _sha256(blob),
            "size_bytes": blob.stat().st_size,
        }
    ]
    assert not (tmp_path / records[0]["path"]).is_symlink()


def test_approved_attempt_ids_keeps_approved_subset_after_complete_review():
    decisions = {
        "animal_approved": {
            "payload": {"decision": "approved_for_lod_and_binding"}
        },
        "animal_rejected": {"payload": {"decision": "rejected"}},
    }
    attempts = {
        "animal_approved": {"instance_id": "animal_approved"},
        "animal_rejected": {"instance_id": "animal_rejected"},
    }

    assert registry.approved_attempt_ids(decisions, attempts) == {
        "animal_approved"
    }


def test_approved_attempt_ids_requires_decisions_for_every_attempt():
    decisions = {
        "animal_approved": {
            "payload": {"decision": "approved_for_lod_and_binding"}
        }
    }
    attempts = {
        "animal_approved": {"instance_id": "animal_approved"},
        "animal_missing": {"instance_id": "animal_missing"},
    }

    with pytest.raises(contracts.ContractError, match="coverage"):
        registry.approved_attempt_ids(decisions, attempts)


def test_approved_attempt_ids_rejects_an_all_rejected_batch():
    decisions = {
        "animal_rejected": {"payload": {"decision": "rejected"}},
    }
    attempts = {
        "animal_rejected": {"instance_id": "animal_rejected"},
    }

    with pytest.raises(contracts.ContractError, match="at least one approved"):
        registry.approved_attempt_ids(decisions, attempts)


def test_frozen_historical_preflight_accepts_additive_current_route_without_rewrite(
    frozen_historical_preflight,
):
    path, _input_dir = frozen_historical_preflight
    before = path.read_bytes()

    preflight, requests, profiles = registry.load_source_contract(
        path, frozen_historical_preflight=True
    )

    assert preflight["preflight_sha256"] == preparation.preflight_sha256(preflight)
    assert len(requests) == 1
    assert len(profiles) == 1
    assert "flux2_pixal3d_static_v1" not in preflight["routes"]
    assert path.read_bytes() == before
    with pytest.raises(
        contracts.ContractError, match="no longer matches current authenticated inputs"
    ):
        registry.load_source_contract(path)


def test_frozen_historical_preflight_rejects_changed_source_file(
    frozen_historical_preflight,
):
    path, input_dir = frozen_historical_preflight
    request_path = input_dir / "instance_requests.json"
    request_path.write_bytes(request_path.read_bytes() + b" ")

    with pytest.raises(contracts.ContractError, match="source file changed"):
        registry.load_source_contract(path, frozen_historical_preflight=True)


def _pixal_identity_fixture(request):
    pixal_input = {
        "path": "/tmp/frozen_pixal_input.png",
        "sha256": "1" * 64,
        "size_bytes": 1,
    }
    controlled = {
        "execution_job_id": f"animal_{request['request_sha256'][:16]}",
        "instance_id": request["instance_id"],
        "request_sha256": request["request_sha256"],
        "generation_seed": request["generation_plan"]["generation_seed"],
        "profile_schema_id": request["profile_schema_id"],
        "profile_sha256": request["profile_sha256"],
        "asset_class": "animal",
        "route": "flux2_pixal3d_animal_v1",
        "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
        "target_physical_profile": copy.deepcopy(
            request["target_physical_profile"]
        ),
        "rig_profile": copy.deepcopy(request["rig_profile"]),
    }
    job = {
        "controlled_request": controlled,
        "legacy_tag": request["instance_id"],
        "seed": request["generation_plan"]["generation_seed"],
        "attempt_ordinal": 0,
        "reference": {"pixal_input": pixal_input},
    }
    attempt = {
        "instance_id": request["instance_id"],
        "execution_job_id": controlled["execution_job_id"],
        "request_sha256": request["request_sha256"],
        "profile_schema_id": request["profile_schema_id"],
        "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
        "target_physical_profile": copy.deepcopy(
            request["target_physical_profile"]
        ),
        "seed": request["generation_plan"]["generation_seed"],
        "attempt_ordinal": 0,
        "pixal_input": pixal_input,
    }
    manifest = {"job_count": 1, "jobs": [job]}
    batch = {
        "status": "passed_generation_and_glb_readback",
        "job_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "attempts": [attempt],
    }
    return manifest, batch


def test_pixal_cross_identity_accepts_exact_canonical_request(
    frozen_historical_preflight,
):
    path, _input_dir = frozen_historical_preflight
    _preflight, requests, _profiles = registry.load_source_contract(
        path, frozen_historical_preflight=True
    )
    request = next(iter(requests.values()))
    manifest, batch = _pixal_identity_fixture(request)

    jobs, attempts = registry.validate_pixal_request_identity(
        batch, manifest, requests
    )

    assert set(jobs) == {request["instance_id"]}
    assert set(attempts) == {request["instance_id"]}


def test_pixal_cross_identity_rejects_attribute_mutation(
    frozen_historical_preflight,
):
    path, _input_dir = frozen_historical_preflight
    _preflight, requests, _profiles = registry.load_source_contract(
        path, frozen_historical_preflight=True
    )
    request = next(iter(requests.values()))
    manifest, batch = _pixal_identity_fixture(request)
    batch["attempts"][0]["sampled_attributes"] = {
        "body_build": "standard",
        "coat_tone": "red",
        "life_stage": "adult",
        "size": "medium",
    }

    with pytest.raises(contracts.ContractError, match="canonical request identity"):
        registry.validate_pixal_request_identity(batch, manifest, requests)


@pytest.fixture
def static_decision_batch(tmp_path):
    instance_id = "dog_fixture_123456789abc"
    review_root = tmp_path / "static_review"
    review_dir = review_root / instance_id
    views_dir = review_dir / "views"
    views_dir.mkdir(parents=True)
    render_manifest = views_dir / "render_manifest.json"
    contracts.write_json_no_replace(render_manifest, {"status": "passed"})
    contact_sheet = review_dir / "contact_sheet.png"
    contact_sheet.write_bytes(b"contact sheet")
    blender_log = review_dir / "blender.log"
    blender_log.write_bytes(b"blender log")
    view_records = {}
    for name in ("front", "back", "side", "top", "quarter"):
        view_path = views_dir / f"{name}.png"
        view_path.write_bytes(f"{name} view".encode())
        view_records[name] = _relative_record(view_path, review_root)
    sampled_attributes = {
        "body_build": "standard",
        "coat_color": "red",
        "size": "medium",
    }
    target_physical_profile = {
        "control_attribute": "size",
        "measurement": "shoulder_height_cm",
    }
    review = {
        "schema": registry.static_decisions.static_reviews.REVIEW_SCHEMA,
        "instance_id": instance_id,
        "request_sha256": "1" * 64,
        "profile_schema_id": "dog_fixture_v1",
        "sampled_attributes": sampled_attributes,
        "target_physical_profile": target_physical_profile,
        "render_manifest": _relative_record(render_manifest, review_root),
        "contact_sheet": _relative_record(contact_sheet, review_root),
        "blender_log": _relative_record(blender_log, review_root),
        "views": view_records,
        "automatic_checks": {"overall": "passed"},
    }
    review["review_sha256"] = _hash_without(review, "review_sha256")
    review_path = review_dir / "static_review_manifest.json"
    contracts.write_json_no_replace(review_path, review)
    review_batch = {
        "schema": registry.static_decisions.static_reviews.REVIEW_BATCH_SCHEMA,
        "status": "rendered_pending_visual_qa",
        "review_count": 1,
        "reviews": [
            {
                "instance_id": instance_id,
                "review_sha256": review["review_sha256"],
                "review": _relative_record(review_path, review_root),
            }
        ],
        "automatic_checks": {"overall": "passed"},
    }
    review_batch["review_batch_sha256"] = _hash_without(
        review_batch, "review_batch_sha256"
    )
    review_batch_path = review_root / "static_review_batch_manifest.json"
    contracts.write_json_no_replace(review_batch_path, review_batch)

    decision_root = tmp_path / "static_decision"
    decision_dir = decision_root / instance_id
    decision_dir.mkdir(parents=True)
    decision = {
        "schema": "avengine_controlled_animal_static_decision_v1",
        "instance_id": instance_id,
        "review_sha256": review["review_sha256"],
        "decision": "approved_for_lod_and_binding",
        "checks": {
            name: True for name in registry.static_decisions.CHECK_FIELDS
        },
        "attribute_evidence": {
            "body_build": "passed_static_visual",
            "coat_color": "passed_static_visual",
            "size": "deferred_to_metric_3d",
        },
        "caveats": [],
        "notes": "Approved fixture.",
        "review": _absolute_record(review_path),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
    }
    decision["decision_sha256"] = _hash_without(decision, "decision_sha256")
    decision_path = decision_dir / "static_decision.json"
    contracts.write_json_no_replace(decision_path, decision)
    decision_batch = {
        "schema": registry.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "static_review_batch": {
            "path": str(review_batch_path.resolve()),
            "sha256": _sha256(review_batch_path),
            "review_batch_sha256": review_batch["review_batch_sha256"],
        },
        "decision_count": 1,
        "approved_count": 1,
        "rejected_count": 0,
        "decisions": [
            {
                "instance_id": instance_id,
                "decision": decision["decision"],
                "decision_sha256": decision["decision_sha256"],
                "record": _relative_record(decision_path, decision_root),
            }
        ],
        "automatic_checks": {"overall": "passed"},
    }
    decision_batch["decision_batch_sha256"] = _hash_without(
        decision_batch, "decision_batch_sha256"
    )
    decision_batch_path = decision_root / "static_decision_batch_manifest.json"
    contracts.write_json_no_replace(decision_batch_path, decision_batch)
    return {
        "batch_path": decision_batch_path,
        "decision_path": decision_path,
        "review_path": review_path,
    }


def test_static_decision_batch_reauthenticates_review_and_semantics(
    static_decision_batch,
):
    _path, payload, records = registry.load_decision_batch(
        static_decision_batch["batch_path"]
    )

    assert payload["approved_count"] == 1
    assert set(records) == {"dog_fixture_123456789abc"}


def test_static_decision_batch_rejects_tampered_review(static_decision_batch):
    static_decision_batch["review_path"].write_bytes(b"tampered review")

    with pytest.raises(contracts.ContractError, match="static review artifact changed"):
        registry.load_decision_batch(static_decision_batch["batch_path"])


def test_static_decision_batch_rejects_duplicate_keys(static_decision_batch):
    path = static_decision_batch["batch_path"]
    encoded = path.read_text(encoding="utf-8")
    path.write_text('{"status":"shadow",' + encoded.lstrip()[1:], encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="duplicate JSON object key"):
        registry.load_decision_batch(path)


def test_static_decision_record_hash_is_checked_before_strict_parse(
    static_decision_batch,
):
    path = static_decision_batch["decision_path"]
    encoded = path.read_text(encoding="utf-8")
    path.write_text('{"decision":"shadow",' + encoded.lstrip()[1:], encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="static decision record changed"):
        registry.load_decision_batch(static_decision_batch["batch_path"])
