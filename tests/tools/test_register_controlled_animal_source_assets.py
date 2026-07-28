import copy
import hashlib
from pathlib import Path

import pytest

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_source_asset_execution as preparation
from tools import register_controlled_animal_source_assets as registry
from tests.tools import (
    test_build_controlled_animal_derived_static_review as derived_review_support,
)


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


def test_registration_replays_current_bounded_repair_contract(tmp_path):
    (
        closure_path,
        repaired,
        source,
    ) = derived_review_support._geometry_fixture(tmp_path)
    closure = contracts.load_json(closure_path)
    repair_path = Path(closure["output"]["repair_manifest"]["path"])
    audit_path = Path(
        closure["output"]["independent_geometry_audit"]["path"]
    )
    review = {
        "derived_geometry": {
            "repair_method": (
                registry.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
            )
        }
    }
    kwargs = {
        "review": review,
        "raw_pixal_path": source["raw_glb"],
        "reviewed_reference_path": source["reference"],
        "raw_decision_path": source["decision_path"],
        "raw_attempt_manifest_path": source["raw_attempt_manifest"],
        "repaired_glb_path": repaired,
        "geometry_closure_path": closure_path,
        "repair_manifest_path": repair_path,
        "geometry_audit_path": audit_path,
    }

    registry._reauthenticate_bounded_derived_geometry(**kwargs)

    legacy = contracts.load_json(repair_path)
    del legacy["implementation_contract"]
    legacy["mutation"] = {
        "mirrored_geometry_source": "same_authenticated_pixal_mesh_only",
        "voxel_resolution": 220,
        "smooth_iterations": 1,
        "target_faces": 100000,
    }
    repair_path.unlink()
    contracts.write_json_no_replace(repair_path, legacy)

    with pytest.raises(
        contracts.ContractError,
        match="current bounded contract.*implementation_contract",
    ):
        registry._reauthenticate_bounded_derived_geometry(**kwargs)


def test_registration_replays_oriented_sheet_contract_and_decision_batch(
    tmp_path,
    monkeypatch,
):
    (
        closure_path,
        repaired,
        source,
    ) = derived_review_support._geometry_fixture(tmp_path)
    closure = contracts.load_json(closure_path)
    repair_path = Path(closure["output"]["repair_manifest"]["path"])
    audit_path = Path(
        closure["output"]["independent_geometry_audit"]["path"]
    )
    decision_batch = tmp_path / "raw_decision_batch.json"
    contracts.write_json_no_replace(
        decision_batch,
        {"decision_batch_sha256": "a" * 64},
    )
    oriented = {
        "implementation_contract": (
            registry.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_manifest": _absolute_record(source["raw_attempt_manifest"]),
            "pixal_source": _absolute_record(source["raw_glb"]),
            "static_decision_batch": _absolute_record(decision_batch),
            "static_decision": _absolute_record(source["decision_path"]),
            "static_decision_value": "approved_for_lod_and_binding",
            "static_decision_state": "research_candidate",
            "raw_four_limbs_usable": True,
            "raw_pose_riggable": True,
        },
        "output": _absolute_record(repaired),
    }
    monkeypatch.setattr(
        registry.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: oriented,
    )
    review = {
        "derived_geometry": {
            "repair_method": (
                registry.derived_review_contract
                .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
            )
        }
    }

    registry._reauthenticate_bounded_derived_geometry(
        review=review,
        raw_pixal_path=source["raw_glb"],
        reviewed_reference_path=source["reference"],
        raw_decision_path=source["decision_path"],
        raw_attempt_manifest_path=source["raw_attempt_manifest"],
        repaired_glb_path=repaired,
        geometry_closure_path=closure_path,
        repair_manifest_path=repair_path,
        geometry_audit_path=audit_path,
        raw_decision_batch_path=decision_batch,
    )


def test_registration_rejects_cross_mode_repair_manifest(
    tmp_path,
    monkeypatch,
):
    (
        closure_path,
        repaired,
        source,
    ) = derived_review_support._geometry_fixture(tmp_path)
    closure = contracts.load_json(closure_path)
    repair_path = Path(closure["output"]["repair_manifest"]["path"])
    audit_path = Path(
        closure["output"]["independent_geometry_audit"]["path"]
    )
    oriented = {
        "implementation_contract": (
            registry.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {},
        "output": _absolute_record(repaired),
    }
    monkeypatch.setattr(
        registry.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: oriented,
    )
    review = {
        "derived_geometry": {
            "repair_method": (
                registry.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
            )
        }
    }

    with pytest.raises(
        contracts.ContractError,
        match="implementation no longer matches",
    ):
        registry._reauthenticate_bounded_derived_geometry(
            review=review,
            raw_pixal_path=source["raw_glb"],
            reviewed_reference_path=source["reference"],
            raw_decision_path=source["decision_path"],
            raw_attempt_manifest_path=source["raw_attempt_manifest"],
            repaired_glb_path=repaired,
            geometry_closure_path=closure_path,
            repair_manifest_path=repair_path,
            geometry_audit_path=audit_path,
        )


def test_registration_rejects_review_raw_decision_batch_path_rebind(tmp_path):
    raw_decision_path = tmp_path / "raw_decision.json"
    raw_decision_path.write_bytes(b"canonical raw decision")
    canonical_batch = tmp_path / "canonical_batch.json"
    canonical_batch.write_bytes(b"canonical batch")
    rebound_batch = tmp_path / "rebound_batch.json"
    rebound_batch.write_bytes(canonical_batch.read_bytes())
    raw_payload = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "decision_sha256": "a" * 64,
    }
    raw_authority = {
        "raw_static_decision": {
            "file": _absolute_record(raw_decision_path),
            **raw_payload,
        },
        "raw_static_decision_batch": {
            "file": _absolute_record(rebound_batch),
            "decision_batch_sha256": "b" * 64,
        },
    }

    with pytest.raises(
        contracts.ContractError,
        match="rebound its canonical raw static authorities",
    ):
        registry._reauthenticate_review_raw_authorities(
            raw_authority=raw_authority,
            raw_decision={
                "path": raw_decision_path,
                "payload": raw_payload,
            },
            decision_batch_path=canonical_batch,
            decision_batch={"decision_batch_sha256": "b" * 64},
        )


def test_register_derived_accepts_canonical_approved_oriented_path(
    tmp_path,
    monkeypatch,
):
    instance_id = "dog_fixture_123456789abc"

    def fixture_file(name, payload=None):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload is None:
            path.write_bytes(name.encode("utf-8"))
        else:
            contracts.write_json_no_replace(path, payload)
        return path

    preflight_path = fixture_file("preflight.json", {"fixture": True})
    pixal_batch_path = fixture_file("pixal_batch.json", {"fixture": True})
    pixal_inputs_path = fixture_file("pixal_inputs.json", {"fixture": True})
    raw_glb = fixture_file("raw.glb")
    raw_attempt = fixture_file("raw_attempt.json", {"fixture": True})
    reference = fixture_file("reference.png")
    pixal_input = fixture_file("pixal_input.png")
    raw_decision_path = fixture_file("raw_decision.json", {"fixture": True})
    decision_batch_path = fixture_file(
        "decision_batch.json", {"fixture": True}
    )
    repaired = fixture_file("repaired.glb")
    closure = fixture_file("closure.json", {"fixture": True})
    repair_manifest = fixture_file(
        "repair_manifest.json", {"fixture": True}
    )
    audit = fixture_file("audit.json", {"fixture": True})
    pbr_contact = fixture_file("pbr_contact.png")
    clay_contact = fixture_file("clay_contact.png")
    review_path = fixture_file("review.json", {"fixture": True})
    derived_decision_path = fixture_file(
        "derived_decision.json", {"fixture": True}
    )
    request = {
        "instance_id": instance_id,
        "profile_schema_id": "dog_fixture_v1",
        "profile_sha256": "1" * 64,
        "request_sha256": "2" * 64,
        "sampled_attributes": {"coat_color": "red", "size": "medium"},
        "target_physical_profile": {"control_attribute": "size"},
        "generation_plan": {"model_revisions": {"pixal": "fixture"}},
    }
    profile = {"profile_schema_id": request["profile_schema_id"]}
    attempt = {
        "execution_job_id": "animal_fixture",
        "output": {
            "path": raw_glb.relative_to(pixal_batch_path.parent).as_posix(),
            "sha256": _sha256(raw_glb),
        },
        "attempt_manifest": {
            "path": raw_attempt.relative_to(
                pixal_batch_path.parent
            ).as_posix(),
            "sha256": _sha256(raw_attempt),
            "size_bytes": raw_attempt.stat().st_size,
        },
        "pixal_input": {"path": str(pixal_input.resolve())},
    }
    input_job = {"reference": {"source": _absolute_record(reference)}}
    raw_payload = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "decision_sha256": "3" * 64,
    }
    raw_decision = {"path": raw_decision_path, "payload": raw_payload}
    review = {
        "review_sha256": "4" * 64,
        "instance_identity": {
            "instance_id": instance_id,
            "profile_schema_id": request["profile_schema_id"],
            "profile_sha256": request["profile_sha256"],
            "request_sha256": request["request_sha256"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
        },
        "source_authorities": {
            "raw_pixal_glb": _absolute_record(raw_glb),
            "reference_2d": _absolute_record(reference),
            "raw_static_decision_batch": {
                "file": _absolute_record(decision_batch_path),
                "decision_batch_sha256": "8" * 64,
            },
            "raw_static_decision": {
                "file": _absolute_record(raw_decision_path),
                **raw_payload,
            },
        },
        "derived_geometry": {
            "repair_method": (
                registry.derived_review_contract
                .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
            ),
            "repaired_glb": _absolute_record(repaired),
            "geometry_closure": _absolute_record(closure),
            "repair_manifest": _absolute_record(repair_manifest),
            "independent_geometry_audit": _absolute_record(audit),
        },
        "evidence": {
            "pbr_five_view": {
                "contact_sheet": {
                    **_absolute_record(pbr_contact),
                    "path": pbr_contact.name,
                }
            },
            "clay_five_view": {
                "contact_sheet": _absolute_record(clay_contact)
            },
        },
    }
    preserved = {
        "decision": raw_payload["decision"],
        "decision_sha256": raw_payload["decision_sha256"],
        "file": _absolute_record(raw_decision_path),
        "formal_dataset_registration_authorized": False,
        "overwritten": False,
        "preserved": True,
        "state_classification": raw_payload["state_classification"],
    }
    derived_decision = {
        "instance_id": instance_id,
        "decision": registry.derived_static_decisions.APPROVED,
        "decision_sha256": "5" * 64,
        "attribute_evidence": {
            "coat_color": "passed_static_visual",
            "size": "deferred_to_metric_3d",
        },
        "authenticated_review_artifact_count": 1,
        "review_binding": {
            "review_file": _absolute_record(review_path),
            "internal_review_sha256": review["review_sha256"],
        },
        "raw_static_decision": preserved,
    }
    monkeypatch.setattr(
        registry,
        "_load_registration_context",
        lambda *_args, **_kwargs: (
            preflight_path,
            {"preflight_sha256": "6" * 64},
            {instance_id: request},
            {request["profile_schema_id"]: profile},
            pixal_batch_path,
            {"batch_sha256": "7" * 64},
            pixal_inputs_path,
            {},
            {instance_id: input_job},
            {instance_id: attempt},
        ),
    )
    monkeypatch.setattr(
        registry,
        "load_decision_batch",
        lambda _path: (
            decision_batch_path,
            {"decision_batch_sha256": "8" * 64},
            {instance_id: raw_decision},
        ),
    )
    monkeypatch.setattr(
        registry.derived_static_decisions,
        "validate_decision",
        lambda _value: copy.deepcopy(derived_decision),
    )
    monkeypatch.setattr(
        registry.derived_review_contract,
        "validate_review",
        lambda _value: copy.deepcopy(review),
    )
    monkeypatch.setattr(
        registry.derived_static_decisions,
        "_authenticate_review_artifacts",
        lambda *_args: 1,
    )
    monkeypatch.setattr(
        registry.derived_static_decisions.stable,
        "_validate_raw_static_decision",
        lambda *_args: copy.deepcopy(preserved),
    )
    replayed = {}

    def replay(**kwargs):
        replayed.update(kwargs)

    monkeypatch.setattr(
        registry,
        "_reauthenticate_bounded_derived_geometry",
        replay,
    )
    monkeypatch.setattr(
        registry,
        "spear_artifact",
        lambda path: {
            "root_id": "spear_repo",
            "path": Path(path).name,
            "sha256": _sha256(Path(path)),
            "size_bytes": Path(path).stat().st_size,
        },
    )
    monkeypatch.setattr(registry, "license_records", lambda: [])
    monkeypatch.setattr(
        registry.contracts,
        "build_source_asset_v2",
        lambda source_request, **kwargs: {
            "schema": contracts.SOURCE_ASSET_SCHEMA,
            "asset_id": source_request["instance_id"],
            "artifacts": kwargs["artifacts"],
        },
    )
    monkeypatch.setattr(
        registry.contracts,
        "validate_source_asset_v2",
        lambda value, **_kwargs: value,
    )
    output_root = tmp_path / "registered"

    manifest_path = registry.register_derived(
        preflight_path,
        pixal_batch_path,
        decision_batch_path,
        derived_decision_path,
        _sha256(derived_decision_path),
        output_root,
    )
    manifest = contracts.load_json(manifest_path)

    assert manifest["schema"] == registry.DERIVED_REGISTRY_SCHEMA
    assert manifest["automatic_checks"] == (
        registry.DERIVED_REGISTRY_AUTOMATIC_CHECKS
    )
    assert replayed["raw_decision_batch_path"] == decision_batch_path
    assert replayed["review"]["derived_geometry"]["repair_method"] == (
        registry.derived_review_contract
        .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
    )


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


def _direct_geometry_source_asset_provenance_fixture(
    frozen_historical_preflight,
):
    preflight_path, _input_dir = frozen_historical_preflight
    _preflight, requests, _profiles = registry.load_source_contract(
        preflight_path,
        frozen_historical_preflight=True,
    )
    request = next(iter(requests.values()))
    execution_job_id = f"direct_{request['request_sha256'][:16]}"
    authority = {
        "instance_id": request["instance_id"],
        "profile_schema_id": request["profile_schema_id"],
        "profile_sha256": request["profile_sha256"],
        "request_sha256": request["request_sha256"],
        "lineage_group_id": request["lineage_group_id"],
        "taxonomy": copy.deepcopy(request["taxonomy"]),
        "fixed_attributes": copy.deepcopy(request["fixed_attributes"]),
        "acoustic_profile": copy.deepcopy(request["acoustic_profile"]),
    }
    models = copy.deepcopy(
        request["generation_plan"]["model_revisions"]
    )
    direct_context = {
        "adopted_batch": {"models": models},
        "attempt": {
            "instance_id": request["instance_id"],
            "execution_job_id": execution_job_id,
            "request_sha256": request["request_sha256"],
            "profile_schema_id": request["profile_schema_id"],
            "sampled_attributes": copy.deepcopy(
                request["sampled_attributes"]
            ),
            "target_physical_profile": copy.deepcopy(
                request["target_physical_profile"]
            ),
        },
        "source_spec": {"instance_id": request["instance_id"]},
        "adoption_context": {
            "controlled": {
                "request_sha256": request["request_sha256"],
                "profile_schema_id": request["profile_schema_id"],
                "rig_profile": copy.deepcopy(request["rig_profile"]),
            }
        },
    }
    artifact = {
        "root_id": "spear_repo",
        "path": "tmp/direct_geometry_fixture.bin",
        "sha256": "8" * 64,
        "size_bytes": 1,
    }
    raw_decision = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "decision_sha256": "9" * 64,
    }
    closure = {
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "manifest_sha256": "a" * 64,
        "inherited_static_judgment": {
            "authority": "canonical_raw_static_approval_v1",
            "decision_sha256": raw_decision["decision_sha256"],
            "new_human_approval_created": False,
        },
    }
    return {
        "authority": authority,
        "direct_context": direct_context,
        "artifacts": {
            "raw_static_decision": copy.deepcopy(artifact),
            "derived_geometry_closure": copy.deepcopy(artifact),
        },
        "raw_decision": raw_decision,
        "closure": closure,
        "execution_job_id": execution_job_id,
        "models": models,
    }


def test_direct_geometry_source_asset_provenance_uses_exact_raw_and_closure(
    frozen_historical_preflight,
):
    case = _direct_geometry_source_asset_provenance_fixture(
        frozen_historical_preflight
    )

    source_asset = registry._build_direct_source_asset_v2(
        authority=case["authority"],
        direct_context=case["direct_context"],
        artifacts=case["artifacts"],
        canonical_raw_decision=case["raw_decision"],
        bounded_geometry_closure=case["closure"],
    )

    assert source_asset["provenance"]["attempt_id"] == (
        f"direct_geometry_{case['execution_job_id']}"
    )
    assert source_asset["provenance"]["models"] == {
        **case["models"],
        registry.DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL: (
            case["raw_decision"]["decision_sha256"]
        ),
        registry.DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL: (
            case["closure"]["manifest_sha256"]
        ),
    }
    assert "derived_static" not in contracts.canonical_json(
        source_asset["provenance"]
    )


def test_direct_geometry_source_asset_provenance_rejects_authority_tamper(
    frozen_historical_preflight,
):
    case = _direct_geometry_source_asset_provenance_fixture(
        frozen_historical_preflight
    )
    case["closure"]["inherited_static_judgment"][
        "decision_sha256"
    ] = "b" * 64

    with pytest.raises(
        contracts.ContractError,
        match="direct geometry provenance authorities changed",
    ):
        registry._build_direct_source_asset_v2(
            authority=case["authority"],
            direct_context=case["direct_context"],
            artifacts=case["artifacts"],
            canonical_raw_decision=case["raw_decision"],
            bounded_geometry_closure=case["closure"],
        )


def _direct_geometry_registration_fixture(
    tmp_path,
    monkeypatch,
    *,
    removed_triangle_count=0,
):
    from tools import (
        publish_generated_animal_geometry_closure as geometry_closures,
    )

    instance_id = "dog_direct_fixture_123456789abc"

    def fixture_file(relative, payload):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            contracts.write_json_no_replace(path, payload)
        return path

    original_raw = fixture_file("original/raw.glb", b"raw Pixel3D bytes")
    original_manifest = fixture_file(
        "original/attempt.json",
        b"original Pixel3D manifest bytes",
    )
    original_input = fixture_file(
        "original/input.png",
        b"original Pixel3D input bytes",
    )
    adopted_root = tmp_path / "adopted"
    adopted_raw = fixture_file(
        "adopted/artifacts/raw.glb",
        original_raw.read_bytes(),
    )
    adopted_manifest = fixture_file(
        "adopted/artifacts/attempt.json",
        original_manifest.read_bytes(),
    )
    adopted_input = fixture_file(
        "adopted/artifacts/input.png",
        original_input.read_bytes(),
    )
    pixal_batch_path = fixture_file("adopted/pixal_batch.json", b"batch")
    source_spec_path = fixture_file("adopted/source_spec.json", b"source spec")
    direct_authority_path = fixture_file(
        "authority/direct_source_authority.json",
        b"direct authority",
    )

    review_root = tmp_path / "raw_static_review"
    review_batch_path = fixture_file(
        "raw_static_review/static_review_batch.json",
        b"static review batch",
    )
    review_path = fixture_file(
        f"raw_static_review/{instance_id}/static_review.json",
        b"static review",
    )
    review_reference = fixture_file(
        f"raw_static_review/{instance_id}/reference.png",
        original_input.read_bytes(),
    )
    review_contact = fixture_file(
        f"raw_static_review/{instance_id}/contact.png",
        b"raw static contact sheet",
    )
    decision_batch_path = fixture_file(
        "raw_static_decision/static_decision_batch.json",
        b"static decision batch",
    )
    decision_path = fixture_file(
        f"raw_static_decision/{instance_id}/static_decision.json",
        b"static decision",
    )
    geometry_closure_path = fixture_file(
        "geometry/geometry_closure.json",
        b"geometry closure",
    )
    repair_manifest_path = fixture_file(
        "geometry/repair_manifest.json",
        b"repair manifest",
    )
    repaired_glb_path = fixture_file(
        "geometry/repaired.glb",
        (
            original_raw.read_bytes()
            if removed_triangle_count == 0
            else b"zero-area triangle filtered bytes"
        ),
    )
    geometry_audit_path = fixture_file(
        "geometry/geometry_audit.json",
        b"geometry audit",
    )
    clay_render_manifest_path = fixture_file(
        "geometry/clay/render_manifest.json",
        b"clay render manifest",
    )
    clay_contact_path = fixture_file(
        "geometry/clay/contact.png",
        b"clay contact sheet",
    )

    attempt = {
        "instance_id": instance_id,
        "execution_job_id": "direct_fixture_job",
        "request_sha256": "1" * 64,
        "profile_schema_id": "dog_direct_fixture_v1",
        "sampled_attributes": {
            "coat_color": "red",
            "size": "medium",
        },
        "target_physical_profile": {
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
        },
        "output": _relative_record(adopted_raw, adopted_root),
        "attempt_manifest": _relative_record(
            adopted_manifest,
            adopted_root,
        ),
        "pixal_input": _relative_record(adopted_input, adopted_root),
    }
    direct_authority = {
        "instance_id": instance_id,
        "profile_schema_id": attempt["profile_schema_id"],
        "profile_sha256": "2" * 64,
        "request_sha256": attempt["request_sha256"],
        "adopted_batch": {"batch_sha256": "3" * 64},
        "authority_sha256": "4" * 64,
    }
    pixal_batch = {
        "batch_sha256": direct_authority["adopted_batch"]["batch_sha256"],
        "models": {},
    }
    direct_context = {
        "adopted_batch_path": pixal_batch_path,
        "adopted_batch": copy.deepcopy(pixal_batch),
        "attempt": attempt,
        "source_spec_path": source_spec_path,
        "source_spec": {"instance_id": instance_id},
        "adoption_context": {
            "bundle_sources": {
                "pixal_raw_glb": original_raw,
                "pixal_attempt_manifest": original_manifest,
                "pixal_input_rgba": original_input,
            }
        },
    }
    checks = {
        name: True for name in registry.static_decisions.CHECK_FIELDS
    }
    raw_payload = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
        "decision_sha256": "5" * 64,
        "checks": checks,
        "attribute_evidence": {
            "coat_color": "passed_static_visual",
            "size": "deferred_to_metric_3d",
        },
    }
    raw_review = {
        "instance_id": instance_id,
        "request_sha256": attempt["request_sha256"],
        "profile_schema_id": attempt["profile_schema_id"],
        "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
        "target_physical_profile": copy.deepcopy(
            attempt["target_physical_profile"]
        ),
        "pixal_output": copy.deepcopy(attempt["output"]),
        "reference_rgba": _relative_record(
            review_reference,
            review_root,
        ),
        "contact_sheet": _relative_record(review_contact, review_root),
    }
    raw_decisions = {
        instance_id: {
            "payload": raw_payload,
            "path": decision_path,
            "static_review": {
                "payload": raw_review,
                "path": review_path,
            },
        }
    }
    decision_batch = {
        "decision_batch_sha256": "6" * 64,
        "static_review_batch": {
            "path": str(review_batch_path.resolve()),
        },
    }
    closure_manifest = {
        "schema": geometry_closures.SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "manifest_sha256": "7" * 64,
        "bounded_repair": {
            "implementation_contract": (
                geometry_closures.oriented_repair.IMPLEMENTATION_CONTRACT
            ),
            "mutation_class": geometry_closures.MUTATION_CLASS,
            "removed_exact_position_degenerate_triangle_count": (
                removed_triangle_count
            ),
            "output_byte_identical_to_raw": (
                removed_triangle_count == 0
            ),
        },
        "inherited_static_judgment": {
            "authority": "canonical_raw_static_approval_v1",
            "decision_sha256": raw_payload["decision_sha256"],
            "checks": copy.deepcopy(checks),
            "inheritance_scope": geometry_closures.INHERITANCE_SCOPE,
            "new_human_approval_created": False,
            "clay_render_human_approval_claimed": False,
        },
        "downstream": {
            "tokenrig_entry_authorized": True,
            "tokenrig_execution_performed": False,
            "formal_dataset_registration_authorized": False,
        },
    }
    closure_replay = {
        "manifest": closure_manifest,
        "paths": {
            "raw_pixal_glb": original_raw,
            "pixal_manifest": original_manifest,
            "source_reference": review_reference,
            "raw_static_decision_batch": decision_batch_path,
            "raw_static_decision": decision_path,
            "raw_static_review": review_path,
            "repair_manifest": repair_manifest_path,
            "repaired_glb": repaired_glb_path,
            "geometry_audit": geometry_audit_path,
            "clay_render_manifest": clay_render_manifest_path,
            "clay_contact_sheet": clay_contact_path,
        },
    }
    closure_call = {}

    def load_closure(_path, **kwargs):
        closure_call.update(kwargs)
        return copy.deepcopy(closure_replay)

    built = {}

    def build_source_asset(
        *,
        authority,
        direct_context,
        artifacts,
        canonical_raw_decision,
        bounded_geometry_closure,
    ):
        built.update(
            {
                "authority": copy.deepcopy(authority),
                "direct_context": copy.deepcopy(direct_context),
                "artifacts": copy.deepcopy(artifacts),
                "canonical_raw_decision": copy.deepcopy(
                    canonical_raw_decision
                ),
                "bounded_geometry_closure": copy.deepcopy(
                    bounded_geometry_closure
                ),
            }
        )
        return {
            "schema": contracts.SOURCE_ASSET_SCHEMA,
            "asset_id": authority["instance_id"],
            "state_classification": "research_candidate",
            "artifacts": copy.deepcopy(dict(artifacts)),
        }

    monkeypatch.setattr(registry, "SPEAR_ROOT", tmp_path)
    monkeypatch.setattr(
        registry.direct_adopter,
        "load_direct_source_authority",
        lambda *_args, **_kwargs: (
            direct_authority_path,
            copy.deepcopy(direct_authority),
            copy.deepcopy(direct_context),
        ),
    )
    monkeypatch.setattr(
        registry.direct_adopter,
        "load_adopted_batch",
        lambda _path: (pixal_batch_path, copy.deepcopy(pixal_batch)),
    )
    monkeypatch.setattr(
        registry,
        "load_decision_batch",
        lambda _path: (
            decision_batch_path,
            copy.deepcopy(decision_batch),
            copy.deepcopy(raw_decisions),
        ),
    )
    monkeypatch.setattr(
        geometry_closures,
        "load_geometry_closure_v2",
        load_closure,
    )
    monkeypatch.setattr(
        registry,
        "_build_direct_source_asset_v2",
        build_source_asset,
    )
    return {
        "instance_id": instance_id,
        "authority_path": direct_authority_path,
        "authority_sha256": _sha256(direct_authority_path),
        "batch_path": pixal_batch_path,
        "decision_batch_path": decision_batch_path,
        "decision_path": decision_path,
        "closure_path": geometry_closure_path,
        "closure_sha256": _sha256(geometry_closure_path),
        "closure_replay": closure_replay,
        "closure_call": closure_call,
        "built": built,
        "attempt": attempt,
        "direct_authority": direct_authority,
        "raw_review": raw_review,
        "original_raw": original_raw,
        "adopted_raw": adopted_raw,
        "repaired_glb": repaired_glb_path,
    }


def _run_direct_geometry_registration(case, tmp_path):
    return registry.register_direct_geometry(
        case["authority_path"],
        case["authority_sha256"],
        case["batch_path"],
        case["decision_batch_path"],
        case["closure_path"],
        case["closure_sha256"],
        tmp_path / "registered",
    )


def test_register_direct_geometry_accepts_byte_copy_bridge_and_noop(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(
        tmp_path,
        monkeypatch,
        removed_triangle_count=0,
    )

    manifest_path = _run_direct_geometry_registration(case, tmp_path)
    manifest = contracts.load_json(manifest_path)

    assert manifest["schema"] == registry.DIRECT_GEOMETRY_REGISTRY_SCHEMA
    assert manifest["formal_dataset_registration_authorized"] is False
    assert "derived_static_decisions" not in manifest
    assert manifest["automatic_checks"] == (
        registry.DIRECT_GEOMETRY_REGISTRY_AUTOMATIC_CHECKS
    )
    assert (
        "raw_owner_static_approval_inherited_without_new_human_claim"
        not in manifest["automatic_checks"]
    )
    assert manifest["automatic_checks"][
        "canonical_raw_static_approval_inherited_without_new_human_claim"
    ] is True
    assert manifest["automatic_checks"][
        "derived_static_decision_not_created"
    ] is True
    assert manifest["source_assets"][0]["attribute_evidence"] == {
        "coat_color": "passed_static_visual",
        "size": "deferred_to_metric_3d",
    }
    assert (
        case["closure_call"]["expected_raw_pixal_glb"]
        == case["original_raw"]
    )
    assert "derived_static_decision" not in case["built"]["artifacts"]
    assert case["built"]["canonical_raw_decision"]["decision_sha256"] == (
        case["closure_replay"]["manifest"]["inherited_static_judgment"][
            "decision_sha256"
        ]
    )
    assert case["built"]["bounded_geometry_closure"]["manifest_sha256"] == (
        case["closure_replay"]["manifest"]["manifest_sha256"]
    )


def test_register_direct_geometry_accepts_zero_area_filter(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(
        tmp_path,
        monkeypatch,
        removed_triangle_count=1,
    )

    manifest_path = _run_direct_geometry_registration(case, tmp_path)

    assert manifest_path.is_file()
    assert _sha256(case["original_raw"]) != _sha256(case["repaired_glb"])


def test_register_direct_geometry_restores_request_bound_physical_provenance(
    tmp_path,
    monkeypatch,
    frozen_historical_preflight,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    _preflight, input_dir = frozen_historical_preflight
    authority_batch = input_dir / "instance_requests.json"
    request = contracts.load_json(authority_batch)["requests"][0]
    target = copy.deepcopy(request["target_physical_profile"])
    reference_provenance = target.pop("reference_provenance")
    case["direct_authority"]["taxonomy"] = copy.deepcopy(request["taxonomy"])
    case["attempt"]["sampled_attributes"] = copy.deepcopy(
        request["sampled_attributes"]
    )
    case["attempt"]["target_physical_profile"] = target
    case["raw_review"]["sampled_attributes"] = copy.deepcopy(
        request["sampled_attributes"]
    )
    case["raw_review"]["target_physical_profile"] = copy.deepcopy(target)

    registry.register_direct_geometry(
        case["authority_path"],
        case["authority_sha256"],
        case["batch_path"],
        case["decision_batch_path"],
        case["closure_path"],
        case["closure_sha256"],
        tmp_path / "registered",
        physical_profile_authority_request_batch_path=authority_batch,
        expected_physical_profile_authority_request_batch_sha256=_sha256(
            authority_batch
        ),
    )

    built = case["built"]
    assert built["direct_context"]["attempt"]["target_physical_profile"][
        "reference_provenance"
    ] == reference_provenance
    assert (
        registry.PHYSICAL_PROFILE_AUTHORITY_ARTIFACT_ROLE
        in built["artifacts"]
    )
    assert built["direct_context"]["adopted_batch"]["models"][
        registry.PHYSICAL_PROFILE_AUTHORITY_REQUEST_MODEL
    ] == request["request_sha256"]


def test_physical_profile_authority_fails_closed_on_sha_or_field_drift(
    frozen_historical_preflight,
):
    _preflight, input_dir = frozen_historical_preflight
    authority_batch = input_dir / "instance_requests.json"
    request = contracts.load_json(authority_batch)["requests"][0]
    target = copy.deepcopy(request["target_physical_profile"])
    target.pop("reference_provenance")
    kwargs = {
        "taxonomy": request["taxonomy"],
        "sampled_attributes": request["sampled_attributes"],
        "target_physical_profile": target,
    }
    with pytest.raises(contracts.ContractError, match="batch changed"):
        registry._authenticate_physical_profile_authority_request_batch(
            authority_batch, "f" * 64, **kwargs
        )
    kwargs["target_physical_profile"]["target_value_cm"] += 1
    with pytest.raises(contracts.ContractError, match="match is not unique"):
        registry._authenticate_physical_profile_authority_request_batch(
            authority_batch, _sha256(authority_batch), **kwargs
        )


def test_register_direct_geometry_rejects_non_copy_adopted_bytes(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    case["adopted_raw"].write_bytes(b"different adopted bytes")
    case["attempt"]["output"].update(
        {
            "sha256": _sha256(case["adopted_raw"]),
            "size_bytes": case["adopted_raw"].stat().st_size,
        }
    )
    case["raw_review"]["pixal_output"] = copy.deepcopy(
        case["attempt"]["output"]
    )

    with pytest.raises(contracts.ContractError, match="is not a byte copy"):
        _run_direct_geometry_registration(case, tmp_path)


def test_register_direct_geometry_rejects_raw_decision_path_rebind(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    rebound = tmp_path / "geometry" / "rebound_decision.json"
    rebound.write_bytes(case["decision_path"].read_bytes())
    case["closure_replay"]["paths"]["raw_static_decision"] = rebound

    with pytest.raises(contracts.ContractError, match="was rebound"):
        _run_direct_geometry_registration(case, tmp_path)


def test_register_direct_geometry_rejects_repair_scope_expansion(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    case["closure_replay"]["manifest"]["bounded_repair"][
        "mutation_class"
    ] = "arbitrary_geometry_rewrite_v1"

    with pytest.raises(
        contracts.ContractError,
        match="research-only inherited approval boundary",
    ):
        _run_direct_geometry_registration(case, tmp_path)


def test_register_direct_geometry_rejects_formal_authorization(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    case["closure_replay"]["manifest"][
        "formal_dataset_registration_authorized"
    ] = True

    with pytest.raises(
        contracts.ContractError,
        match="research-only inherited approval boundary",
    ):
        _run_direct_geometry_registration(case, tmp_path)


def test_register_direct_geometry_rejects_false_noop_claim(
    tmp_path,
    monkeypatch,
):
    case = _direct_geometry_registration_fixture(tmp_path, monkeypatch)
    case["repaired_glb"].write_bytes(b"silently rewritten geometry")

    with pytest.raises(
        contracts.ContractError,
        match="repair bytes contradict",
    ):
        _run_direct_geometry_registration(case, tmp_path)


def test_main_rejects_derived_and_direct_geometry_modes_together(
    tmp_path,
    capsys,
):
    result = registry.main(
        [
            "--direct-source-authority",
            str(tmp_path / "authority.json"),
            "--expected-direct-source-authority-sha256",
            "1" * 64,
            "--pixal-batch",
            str(tmp_path / "batch.json"),
            "--static-decision-batch",
            str(tmp_path / "decision_batch.json"),
            "--derived-static-decision",
            str(tmp_path / "derived.json"),
            "--expected-derived-static-decision-sha256",
            "2" * 64,
            "--geometry-closure",
            str(tmp_path / "closure.json"),
            "--expected-geometry-closure-sha256",
            "3" * 64,
            "--output-root",
            str(tmp_path / "registered"),
        ]
    )

    assert result == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_rejects_incomplete_geometry_closure_hash_pair(
    tmp_path,
    capsys,
):
    result = registry.main(
        [
            "--direct-source-authority",
            str(tmp_path / "authority.json"),
            "--expected-direct-source-authority-sha256",
            "1" * 64,
            "--pixal-batch",
            str(tmp_path / "batch.json"),
            "--static-decision-batch",
            str(tmp_path / "decision_batch.json"),
            "--geometry-closure",
            str(tmp_path / "closure.json"),
            "--output-root",
            str(tmp_path / "registered"),
        ]
    )

    assert result == 2
    assert "geometry closure path and expected SHA-256" in capsys.readouterr().err
