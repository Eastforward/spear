from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from tools import build_controlled_animal_derived_static_review as producer
from tools import controlled_animal_derived_static_review_contract as contract
from tools import controlled_source_asset_schema as contracts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict:
    recorded = path if relative_to is None else path.relative_to(relative_to)
    return {
        "path": str(recorded),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _file(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_json_no_replace(path, payload)
    return path


def _png(path: Path, color=(80, 100, 120, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (480, 480), color).save(path, format="PNG")
    return path


def _rehash(review: dict) -> dict:
    review["review_sha256"] = contract.hash_without(review, "review_sha256")
    return review


def _valid_review(tmp_path: Path) -> dict:
    external = tmp_path / "external"
    internal = tmp_path / "review"
    files = {
        name: _file(external / f"{name}.bin", name.encode())
        for name in (
            "preflight",
            "pixal_batch",
            "decision_batch",
            "decision",
            "raw_pixal",
            "reference",
            "closure",
            "repaired",
            "repair_manifest",
            "geometry_audit",
            "tool",
            "renderer",
        )
    }
    clay_render = _json(external / "clay_render.json", {"mode": "clay"})
    clay_views = {
        name: _png(external / "clay" / f"{name}.png") for name in contract.VIEWS
    }
    clay_contact = _png(external / "clay_contact.png")
    pbr_render = _json(internal / "pbr/render.json", {"mode": "pbr"})
    pbr_views = {
        name: _png(internal / "pbr" / f"{name}.png") for name in contract.VIEWS
    }
    pbr_contact = _png(internal / "pbr/contact.png")
    pbr_log = _file(internal / "pbr/blender.log", b"Blender fixture")
    review = {
        "schema": contract.REVIEW_SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": contract.REVIEW_STATUS,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "claim_boundary": "Pending-human evidence only.",
        "instance_identity": {
            "instance_id": "cat_fixture_123456789abc",
            "profile_schema_id": "cat_fixture_v1",
            "profile_sha256": "1" * 64,
            "request_sha256": "2" * 64,
            "taxonomy": {"species": "cat", "breed": "british_shorthair"},
            "fixed_attributes": {"tail_shape": "thick_medium"},
            "sampled_attributes": {"coat_color": "blue"},
            "target_physical_profile": {"control_attribute": "size"},
        },
        "source_authorities": {
            "frozen_preflight": {
                "file": _record(files["preflight"]),
                "preflight_sha256": "3" * 64,
                "validation_mode": "frozen_historical_preflight_v1",
            },
            "pixal_batch": {
                "file": _record(files["pixal_batch"]),
                "batch_sha256": "4" * 64,
            },
            "raw_static_decision_batch": {
                "file": _record(files["decision_batch"]),
                "decision_batch_sha256": "5" * 64,
            },
            "raw_static_decision": {
                "file": _record(files["decision"]),
                "decision_sha256": "6" * 64,
                "decision": "rejected",
                "state_classification": "rejected",
                "formal_dataset_registration_authorized": False,
            },
            "raw_pixal_glb": _record(files["raw_pixal"]),
            "reference_2d": _record(files["reference"]),
        },
        "derived_geometry": {
            "geometry_closure": _record(files["closure"]),
            "repaired_glb": _record(files["repaired"]),
            "repair_manifest": _record(files["repair_manifest"]),
            "independent_geometry_audit": _record(files["geometry_audit"]),
            "repair_method": "bounded_same_source_fixture",
            "lineage_kind": "bounded_same_pixal_mesh_repair",
            "automatic_gate_statuses": {
                "four_independent_leg_chains": "passed",
                "no_low_cross_limb_membrane": "passed",
                "nonmanifold": "passed",
                "watertight": "passed",
            },
            "inherited_manual_review_statuses": {
                "single_breed_valid_tail": "passed_manual_multiview",
                "centerline": "passed_manual_top_view_review",
                "clay_five_view": "passed_manual_geometry_review",
            },
            "pbr_container_readback": {
                "material_count": 1,
                "texture_count": 2,
                "image_count": 2,
                "pbr_fidelity_qualified": False,
            },
        },
        "evidence": {
            "clay_five_view": {
                "front_axis": "negative-x",
                "resolution": [480, 480],
                "material_mode": "neutral_clay_geometry_qa_v1",
                "render_manifest": _record(clay_render),
                "views": {name: _record(path) for name, path in clay_views.items()},
                "contact_sheet": _record(clay_contact),
            },
            "pbr_five_view": {
                "front_axis": "negative-x",
                "resolution": [480, 480],
                "material_mode": "ue_animal_nonmetallic_roughness_preview_v1",
                "render_manifest": _record(pbr_render, relative_to=internal),
                "views": {
                    name: _record(path, relative_to=internal)
                    for name, path in pbr_views.items()
                },
                "contact_sheet": _record(pbr_contact, relative_to=internal),
                "execution_log": _record(pbr_log, relative_to=internal),
            },
        },
        "producer": {
            "tool": _record(files["tool"]),
            "renderer": _record(files["renderer"]),
            "blender": {
                "path": "/opt/blender/blender",
                "version": "4.2.1 LTS",
                "build_hash": "396f546c9d82",
            },
        },
        "automatic_checks": {
            name: True for name in sorted(contract.AUTOMATIC_CHECK_FIELDS)
        },
        "human_review": {
            "status": "pending",
            "decision": None,
            "checks": {name: None for name in sorted(contract.HUMAN_CHECK_FIELDS)},
            "review_authority_required": (
                "explicit_project_owner_decision_bound_to_review_file_sha256"
            ),
            "decision_artifact": None,
        },
        "next_gate": contract.NEXT_GATE,
    }
    return _rehash(review)


def test_contract_accepts_only_pending_human_review(tmp_path):
    review = _valid_review(tmp_path)

    assert contract.validate_review(review) == review


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(
                {"formal_dataset_registration_authorized": True}
            ),
            "cannot authorize formal",
        ),
        (
            lambda value: value["human_review"].update({"decision": "approved"}),
            "must remain explicitly pending",
        ),
        (
            lambda value: value["human_review"]["checks"].update(
                {"one_tail_without_duplicate_tail": True}
            ),
            "cannot be inferred",
        ),
        (
            lambda value: value["source_authorities"]["raw_static_decision"].update(
                {
                    "decision": "approved_for_lod_and_binding",
                    "state_classification": "research_candidate",
                }
            ),
            "raw static rejection",
        ),
        (
            lambda value: value["derived_geometry"]["pbr_container_readback"].update(
                {"pbr_fidelity_qualified": True}
            ),
            "PBR fidelity",
        ),
        (
            lambda value: value["automatic_checks"].update(
                {"raw_static_rejection_preserved": False}
            ),
            "automatic checks",
        ),
    ],
)
def test_contract_rejects_authority_upgrades(tmp_path, mutate, message):
    review = _valid_review(tmp_path)
    mutate(review)
    _rehash(review)

    with pytest.raises(contract.DerivedStaticReviewContractError, match=message):
        contract.validate_review(review)


def test_contract_rejects_self_hash_change(tmp_path):
    review = _valid_review(tmp_path)
    review["claim_boundary"] += " changed"

    with pytest.raises(contract.DerivedStaticReviewContractError, match="self-hash"):
        contract.validate_review(review)


def test_regular_file_rejects_leaf_symlink(tmp_path):
    target = _file(tmp_path / "target.bin")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(target)

    with pytest.raises(contracts.ContractError, match="unsafe symlink"):
        producer._regular_file(alias, "fixture")


def test_new_output_root_rejects_arbitrary_symlink_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="unsafe symlink component"):
        producer._new_output_root(alias / "review")

    assert not (real / "review").exists()


def test_new_output_root_rejects_existing_path_without_changing_it(tmp_path):
    output = tmp_path / "review"
    output.mkdir()
    sentinel = _file(output / "sentinel", b"preserve")

    with pytest.raises(contracts.ContractError, match="refusing to replace output"):
        producer._new_output_root(output)

    assert sentinel.read_bytes() == b"preserve"


def _geometry_fixture(tmp_path: Path):
    raw = _file(tmp_path / "raw.glb", b"raw glb")
    raw_manifest = _json(tmp_path / "raw.manifest.json", {"backend": "pixal3d"})
    reference = _png(tmp_path / "reference.png")
    decision = _json(tmp_path / "raw_decision.json", {"decision": "rejected"})
    repaired = _file(tmp_path / "repaired.glb", b"repaired glb")
    repair_manifest = {
        "schema": "avengine_pixal_same_mesh_mirrored_limb_repair_v1",
        "formal_dataset_registration_authorized": False,
        "lineage": {
            "instance_id": "cat_fixture_123456789abc",
            "approved_reference": _record(reference),
            "pixal_manifest": _record(raw_manifest),
            "pixal_source": _record(raw),
            "static_decision": _record(decision),
            "static_decision_state": "rejected",
            "raw_four_limbs_usable": False,
            "raw_pose_riggable": False,
        },
        "output": _record(repaired),
        "checks": {
            "authenticated_same_pixal_mesh_only": True,
            "near_side_front_and_hind_donor_masks_passed": True,
            "tail_region_never_selected_for_replacement_or_mirroring": True,
            "single_tail_preserved_by_protected_edit_scope": True,
            "four_independent_low_limb_chains": True,
            "no_low_cross_limb_membrane": True,
            "one_connected_output_component": True,
            "watertight_manifold_topology": True,
            "output_within_authenticated_source_envelope": True,
        },
    }
    repair_manifest_path = _json(tmp_path / "repair_manifest.json", repair_manifest)
    geometry_audit = {
        "schema": "avengine_quadruped_i23d_geometry_audit_v3",
        "records": [
            {
                "mesh": {
                    "absolute_path": str(repaired),
                    "sha256": _sha256(repaired),
                    "size_bytes": repaired.stat().st_size,
                }
            }
        ],
    }
    geometry_audit_path = _json(tmp_path / "geometry_audit.json", geometry_audit)
    clay_root = tmp_path / "clay"
    clay_views = {name: _png(clay_root / f"{name}.png") for name in contract.VIEWS}
    clay_render = {
        "input": str(repaired),
        "front_axis": "negative-x",
        "resolution": [480, 480],
        "views": {name: [0, 0, 0] for name in contract.VIEWS},
        "material_preview": {"mode": "neutral_clay_geometry_qa_v1"},
    }
    clay_render_path = _json(clay_root / "render_manifest.json", clay_render)
    clay_contact = _png(tmp_path / "clay_contact.png")
    closure = {
        "schema": "avengine_generated_animal_geometry_closure_v1",
        "status": "pass_geometry_only",
        "candidate": {
            "instance_id": "cat_fixture_123456789abc",
            "source_pixal_glb": _record(raw),
            "owner_approved_flux_reference": {
                "path": str(reference),
                "sha256": _sha256(reference),
            },
        },
        "repair": {
            "method": "bounded_same_source_fixture",
            "same_source_surface_only": True,
            "external_geometry_inputs": [],
            "external_skeleton_inputs": [],
            "external_weight_inputs": [],
            "external_material_inputs": [],
            "external_texture_inputs": [],
            "animation_inputs": [],
            "tail_region_selected_for_replacement_or_mirroring": False,
        },
        "output": {
            "glb": _record(repaired),
            "repair_manifest": {
                "path": str(repair_manifest_path),
                "sha256": _sha256(repair_manifest_path),
            },
            "independent_geometry_audit": {
                "path": str(geometry_audit_path),
                "sha256": _sha256(geometry_audit_path),
            },
        },
        "gates": {
            "four_independent_leg_chains": {"status": "passed"},
            "no_low_cross_limb_membrane": {"status": "passed"},
            "nonmanifold": {"status": "passed"},
            "watertight": {"status": "passed"},
            "single_breed_valid_tail": {"status": "passed_manual_multiview"},
            "centerline": {"status": "passed_manual_top_view_review"},
            "five_view_readback": {
                "status": "passed_manual_geometry_review",
                "resolution": [480, 480],
                "material_mode": "neutral_clay_geometry_qa_v1",
                "checks": {"one_tail": True, "four_limbs": True},
                "render_manifest": {
                    "path": str(clay_render_path),
                    "sha256": _sha256(clay_render_path),
                },
                "views": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in clay_views.items()
                },
                "contact_sheet": _record(clay_contact),
            },
        },
        "container_readback": {
            "material_count": 1,
            "texture_count": 2,
            "image_count": 2,
            "skin_count": 0,
            "animation_count": 0,
            "pbr_fidelity_qualified": False,
        },
        "downstream": {
            "ue_import_executed": False,
            "formal_dataset_registration_authorized": False,
        },
    }
    closure_path = _json(tmp_path / "closure.json", closure)
    source = {
        "request": {
            "instance_id": "cat_fixture_123456789abc",
        },
        "raw_glb": raw,
        "raw_attempt_manifest": raw_manifest,
        "reference": reference,
        "decision_path": decision,
    }
    return closure_path, repaired, source


def test_geometry_closure_preserves_raw_rejection_and_pending_pbr(tmp_path):
    closure_path, repaired, source = _geometry_fixture(tmp_path)

    result = producer._validate_geometry_closure(
        closure_path=closure_path,
        repaired_glb=repaired,
        source=source,
    )

    assert result["automatic_statuses"]["watertight"] == "passed"
    assert result["pbr_container"]["pbr_fidelity_qualified"] is False
    assert result["clay"]["front_axis"] == "negative-x"


def test_geometry_closure_rejects_raw_decision_upgrade(tmp_path):
    closure_path, repaired, source = _geometry_fixture(tmp_path)
    repair_path = Path(
        contracts.load_json(closure_path)["output"]["repair_manifest"]["path"]
    )
    repair = contracts.load_json(repair_path)
    repair["lineage"]["static_decision_state"] = "approved_for_lod_and_binding"
    repair_path.unlink()
    _json(repair_path, repair)
    closure = contracts.load_json(closure_path)
    closure["output"]["repair_manifest"]["sha256"] = _sha256(repair_path)
    closure_path.unlink()
    _json(closure_path, closure)

    with pytest.raises(contracts.ContractError, match="upgraded the raw rejection"):
        producer._validate_geometry_closure(
            closure_path=closure_path,
            repaired_glb=repaired,
            source=source,
        )


def test_geometry_closure_rejects_pbr_qualification_without_human(tmp_path):
    closure_path, repaired, source = _geometry_fixture(tmp_path)
    closure = contracts.load_json(closure_path)
    closure["container_readback"]["pbr_fidelity_qualified"] = True
    closure_path.unlink()
    _json(closure_path, closure)

    with pytest.raises(contracts.ContractError, match="PBR container"):
        producer._validate_geometry_closure(
            closure_path=closure_path,
            repaired_glb=repaired,
            source=source,
        )


def test_publish_review_outputs_no_decision_registry_or_ue_job(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    request = {
        "instance_id": "cat_fixture_123456789abc",
        "profile_schema_id": "cat_fixture_v1",
        "profile_sha256": "1" * 64,
        "request_sha256": "2" * 64,
        "taxonomy": {"species": "cat", "breed": "british_shorthair"},
        "fixed_attributes": {"tail_shape": "thick_medium"},
        "sampled_attributes": {"coat_color": "blue"},
        "target_physical_profile": {"control_attribute": "size"},
    }
    source_files = {
        name: _file(source_root / f"{name}.bin", name.encode())
        for name in (
            "preflight",
            "pixal_batch",
            "decision_batch",
            "decision",
            "raw",
            "reference",
            "repair_manifest",
            "audit",
            "clay_render",
            "clay_contact",
        )
    }
    clay_views = {
        name: _png(source_root / "clay" / f"{name}.png") for name in contract.VIEWS
    }
    source = {
        "preflight_path": source_files["preflight"],
        "preflight": {"preflight_sha256": "3" * 64},
        "request": request,
        "pixal_batch_path": source_files["pixal_batch"],
        "pixal_batch": {"batch_sha256": "4" * 64},
        "decision_batch_path": source_files["decision_batch"],
        "decision_batch": {"decision_batch_sha256": "5" * 64},
        "decision_path": source_files["decision"],
        "decision": {"decision_sha256": "6" * 64},
        "raw_glb": source_files["raw"],
        "reference": source_files["reference"],
    }
    repaired = _file(source_root / "repaired.glb", b"repaired")
    closure = _file(source_root / "closure.json", b"closure")
    geometry = {
        "closure": {"repair": {"method": "bounded_same_source_fixture"}},
        "repair_manifest_path": source_files["repair_manifest"],
        "geometry_audit_path": source_files["audit"],
        "automatic_statuses": {
            "four_independent_leg_chains": "passed",
            "no_low_cross_limb_membrane": "passed",
            "nonmanifold": "passed",
            "watertight": "passed",
        },
        "inherited_statuses": {
            "single_breed_valid_tail": "passed_manual_multiview",
            "centerline": "passed_manual_top_view_review",
            "clay_five_view": "passed_manual_geometry_review",
        },
        "clay": {
            "front_axis": "negative-x",
            "render_manifest_path": source_files["clay_render"],
            "views": clay_views,
            "contact_sheet": source_files["clay_contact"],
        },
        "pbr_container": {
            "material_count": 1,
            "texture_count": 2,
            "image_count": 2,
            "pbr_fidelity_qualified": False,
        },
    }

    monkeypatch.setattr(
        producer,
        "_authenticate_source_authorities",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        producer,
        "_external_file",
        lambda path, _expected, _label: Path(path).resolve(),
    )
    monkeypatch.setattr(
        producer,
        "_validate_geometry_closure",
        lambda **_kwargs: geometry,
    )

    def fake_pbr_review(**kwargs):
        staging = kwargs["staging"]
        root = staging / "pbr"
        render = _json(root / "render.json", {"mode": "pbr"})
        views = {name: _png(root / f"{name}.png") for name in contract.VIEWS}
        contact = _png(root / "contact.png")
        log = _file(root / "blender.log", b"log")
        evidence = {
            "front_axis": "negative-x",
            "resolution": [480, 480],
            "material_mode": "ue_animal_nonmetallic_roughness_preview_v1",
            "render_manifest": _record(render, relative_to=staging),
            "views": {
                name: _record(path, relative_to=staging) for name, path in views.items()
            },
            "contact_sheet": _record(contact, relative_to=staging),
            "execution_log": _record(log, relative_to=staging),
        }
        return evidence, {
            "path": "/opt/blender/blender",
            "version": "4.2.1 LTS",
            "build_hash": "396f546c9d82",
        }

    monkeypatch.setattr(producer, "_run_pbr_review", fake_pbr_review)
    monkeypatch.setattr(
        producer,
        "RENDERER",
        _file(source_root / "renderer.py", b"renderer"),
    )
    monkeypatch.setattr(
        producer, "__file__", str(_file(source_root / "tool.py", b"tool"))
    )
    output = tmp_path / "output"

    manifest_path = producer.publish_review(
        instance_id=request["instance_id"],
        preflight_path=source_files["preflight"],
        expected_preflight_sha256="a" * 64,
        pixal_batch_path=source_files["pixal_batch"],
        expected_pixal_batch_sha256="b" * 64,
        raw_static_decision_batch_path=source_files["decision_batch"],
        expected_raw_static_decision_batch_sha256="c" * 64,
        geometry_closure_path=closure,
        expected_geometry_closure_sha256="d" * 64,
        repaired_glb_path=repaired,
        expected_repaired_glb_sha256="e" * 64,
        output_root=output,
    )

    payload = contract.validate_review(contracts.load_json(manifest_path))
    assert payload["human_review"]["decision"] is None
    assert all(value is None for value in payload["human_review"]["checks"].values())
    assert payload["formal_dataset_registration_authorized"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "derived_static_review.json",
        "pbr",
    ]
    assert not list(output.rglob("*decision*"))
    assert not list(output.rglob("*registry*"))
    assert not list(output.rglob("*ue_import*"))
    assert (output.stat().st_mode & 0o222) == 0


def test_cli_requires_all_external_authority_hashes():
    parser = producer.build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--instance-id",
                "cat_fixture",
                "--preflight",
                "preflight.json",
                "--pixal-batch",
                "pixal.json",
                "--raw-static-decision-batch",
                "decision.json",
                "--geometry-closure",
                "closure.json",
                "--repaired-glb",
                "repaired.glb",
                "--output-root",
                "output",
            ]
        )
