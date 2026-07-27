from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess

from PIL import Image
import pytest

from tools import audit_mesh_efficiency
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import review_controlled_static_object_candidates as decision_review
from tools import run_controlled_animal_pixal_jobs as pixal_runner
from tools import run_controlled_static_object_reviews as reviews


INSTANCE_ID = "kitchen_appliance_stovetop_kettle_test_0001"
REQUEST_SHA256 = "1" * 64
PROFILE_SHA256 = "2" * 64
PROFILE_ID = "kitchen_appliance_stovetop_kettle_product_view_v1"


def _record(path: Path, *, root: Path | None = None) -> dict:
    path = path.resolve()
    return {
        "path": (
            path.relative_to(root.resolve()).as_posix()
            if root is not None
            else str(path)
        ),
        "sha256": reviews._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contracts.canonical_json(payload) + "\n", encoding="utf-8")


def _write_test_glb(path: Path, *, skins: int = 0, animations: int = 0) -> None:
    document = {
        "asset": {"version": "2.0"},
        "accessors": [{"count": 3}, {"count": 3}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [{}],
        "textures": [{"source": 0}],
        "images": [{"uri": "data:image/png;base64,AA=="}],
        "skins": [{} for _index in range(skins)],
        "animations": [{} for _index in range(animations)],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total = 12 + 8 + len(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )


def _target_physical_profile() -> dict:
    return {
        "profile_id": "kettle_physical_candidate_v1",
        "control_attribute": None,
        "selected_value": "fixed",
        "measurement": "height_cm",
        "mode": "absolute_measurement",
        "reference_value_cm": 24,
        "reference_provenance": {
            "status": "provisional",
            "source_id": "fixture",
            "artifact": None,
            "notes": "fixture",
        },
        "target_value_cm": 24,
        "tolerance_cm": 5,
    }


def _static_pixal_fixture(tmp_path: Path) -> dict:
    input_root = tmp_path / "pixal_inputs"
    output_root = tmp_path / "pixal_outputs"
    reference = (
        input_root / "segmentation" / INSTANCE_ID / "input_rgba_isnet.png"
    )
    reference.parent.mkdir(parents=True)
    Image.new("RGBA", (64, 64), (180, 50, 40, 255)).save(reference)

    glb = output_root / INSTANCE_ID / "pixal_raw_1024.glb"
    _write_test_glb(glb)
    target = _target_physical_profile()
    controlled = {
        "execution_job_id": f"static_{REQUEST_SHA256[:16]}",
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "generation_seed": 41,
        "profile_schema_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "asset_class": reviews.STATIC_ASSET_CLASS,
        "route": reviews.STATIC_ROUTE,
        "sampled_attributes": {"body_color": "red"},
        "target_physical_profile": target,
        "rig_profile": None,
    }
    job = {
        "legacy_tag": INSTANCE_ID,
        "candidate_tag": f"{INSTANCE_ID}_pixal_v1",
        "asset_class": reviews.STATIC_ASSET_CLASS,
        "route": reviews.STATIC_ROUTE,
        "seed": 41,
        "attempt_ordinal": 0,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "reference": {
            "source": _record(reference),
            "pixal_input": _record(reference),
            "normalization": "pinned_isnet_general_use_alpha_v1",
        },
        "output": str(glb.resolve()),
        "manifest": str(glb.with_suffix(".manifest.json").resolve()),
        "controlled_request": controlled,
        "model_revisions": {
            "pixal3d": pixal_inputs.PIXAL_MODEL_REVISION,
            "dino": pixal_inputs.DINO_REVISION,
        },
        "parameters": {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": False,
        },
    }
    input_payload = {
        "schema": pixal_inputs.PIXAL_INPUT_SCHEMA,
        "status": "ready_for_pixal3d",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "asset_class": reviews.STATIC_ASSET_CLASS,
        "route": reviews.STATIC_ROUTE,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "upstream_flux_one_shot_evidence": {
            "mode": "native_policy_enforced_before_inference",
            "policy": one_shot.policy_record(),
            "flux_batch_sha256": "3" * 64,
            "profile_qualification_authorized": True,
        },
        "pixal_output_root": str(output_root.resolve()),
        "job_count": 1,
        "jobs": [job],
        "automatic_checks": {
            "static_jobs_have_no_rig_or_animation_binding": True,
            "overall": "passed",
        },
    }
    input_payload["manifest_sha256"] = reviews._json_sha256(input_payload)
    input_manifest = input_root / "pixal_inputs_manifest.json"
    _write_json(input_manifest, input_payload)

    attempt_manifest = {
        "backend": "pixal3d",
        "output": {
            "path": str(glb.resolve()),
            "sha256": reviews._sha256_file(glb),
            "bytes": glb.stat().st_size,
        },
        "controlled_request": controlled,
    }
    attempt_manifest_path = glb.with_suffix(".manifest.json")
    _write_json(attempt_manifest_path, attempt_manifest)
    live = audit_mesh_efficiency.mesh_stats(glb)
    mesh_readback = {
        key: value for key, value in live.items() if key not in {"path", "exists"}
    }
    attempt = {
        "instance_id": INSTANCE_ID,
        "execution_job_id": controlled["execution_job_id"],
        "request_sha256": REQUEST_SHA256,
        "profile_schema_id": PROFILE_ID,
        "sampled_attributes": {"body_color": "red"},
        "target_physical_profile": target,
        "gpu": 0,
        "seed": 41,
        "attempt_ordinal": 0,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "pixal_input": _record(reference),
        "output": _record(glb, root=output_root),
        "attempt_manifest": _record(attempt_manifest_path, root=output_root),
        "mesh_readback": mesh_readback,
        "status": "passed_generation_and_glb_readback",
        "next_gate": "static_visual_qa",
    }
    batch = {
        "schema": pixal_runner.BATCH_SCHEMA,
        "status": "passed_generation_and_glb_readback",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "pixal_inputs": {
            "path": str(input_manifest.resolve()),
            "sha256": reviews._sha256_file(input_manifest),
            "manifest_sha256": input_payload["manifest_sha256"],
        },
        "job_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "attempts": [attempt],
        "automatic_checks": {"overall": "passed"},
    }
    batch["batch_sha256"] = reviews._hash_without(batch, "batch_sha256")
    batch_path = output_root / "pixal_batch_manifest.json"
    _write_json(batch_path, batch)
    return {
        "batch_path": batch_path,
        "batch": batch,
        "input_manifest": input_manifest,
        "input_payload": input_payload,
        "reference": reference,
        "glb": glb,
    }


def _reseal_input_and_batch(fixture: dict) -> None:
    input_payload = fixture["input_payload"]
    input_payload["manifest_sha256"] = reviews._hash_without(
        input_payload, "manifest_sha256"
    )
    _write_json(fixture["input_manifest"], input_payload)
    batch = fixture["batch"]
    batch["pixal_inputs"] = {
        "path": str(fixture["input_manifest"].resolve()),
        "sha256": reviews._sha256_file(fixture["input_manifest"]),
        "manifest_sha256": input_payload["manifest_sha256"],
    }
    batch["batch_sha256"] = reviews._hash_without(batch, "batch_sha256")
    _write_json(fixture["batch_path"], batch)


def _fake_renderer(command, **_kwargs):
    output_dir = Path(command[command.index("--output-dir") + 1])
    input_path = Path(command[command.index("--input") + 1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    clay = "--clay-preview" in command
    for index, view in enumerate(reviews.RENDER_VIEWS):
        Image.new("RGB", (480, 480), (40 + index * 20, 80, 120)).save(
            output_dir / f"{view}.png"
        )
    _write_json(
        output_dir / "render_manifest.json",
        {
            "input": str(input_path),
            "front_axis": "negative-y",
            "views": {
                view: [index, 0, 0]
                for index, view in enumerate(reviews.RENDER_VIEWS)
            },
            "resolution": [480, 480],
            "material_preview": {
                "mode": reviews.CLAY_MODE if clay else reviews.RAW_PBR_MODE
            },
        },
    )
    return subprocess.CompletedProcess(command, 0)


def test_load_static_pixal_batch_reauthenticates_route_rig_and_mesh(tmp_path):
    fixture = _static_pixal_fixture(tmp_path)

    path, batch, bindings = reviews.load_static_pixal_batch(fixture["batch_path"])

    assert path == fixture["batch_path"].resolve()
    assert batch["formal_dataset_registration_authorized"] is False
    assert set(bindings) == {INSTANCE_ID}
    binding = bindings[INSTANCE_ID]
    assert binding["controlled_request"]["asset_class"] == "static_object"
    assert binding["controlled_request"]["route"] == reviews.STATIC_ROUTE
    assert binding["controlled_request"]["rig_profile"] is None
    assert binding["mesh_readback"]["skins"] == 0
    assert binding["mesh_readback"]["animations"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("asset_class", "animal", "static-object route"),
        ("route", "flux2_pixal3d_animal_v1", "static-object route"),
        ("rig_mode", "animated_transfer", "rig binding"),
    ],
)
def test_static_pixal_input_semantic_tampering_fails_closed(
    tmp_path, field, value, message
):
    fixture = _static_pixal_fixture(tmp_path)
    fixture["input_payload"]["jobs"][0][field] = value
    _reseal_input_and_batch(fixture)

    with pytest.raises(contracts.ContractError, match=message):
        reviews.load_static_pixal_batch(fixture["batch_path"])


@pytest.mark.parametrize("field", ["skins", "animations"])
def test_static_pixal_mesh_rigging_claims_fail_closed(tmp_path, field):
    fixture = _static_pixal_fixture(tmp_path)
    fixture["batch"]["attempts"][0]["mesh_readback"][field] = 1
    fixture["batch"]["batch_sha256"] = reviews._hash_without(
        fixture["batch"], "batch_sha256"
    )
    _write_json(fixture["batch_path"], fixture["batch"])

    with pytest.raises(contracts.ContractError, match="mesh readback changed"):
        reviews.load_static_pixal_batch(fixture["batch_path"])


def test_run_reviews_publishes_reference_raw_pbr_and_deferred_heading(
    tmp_path, monkeypatch
):
    fixture = _static_pixal_fixture(tmp_path)
    fake_blender = tmp_path / "blender"
    fake_blender.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(reviews, "BLENDER", fake_blender)
    monkeypatch.setattr(reviews.subprocess, "run", _fake_renderer)
    monkeypatch.setattr(reviews.immutable, "_seal_readonly_tree", lambda _path: None)
    output_root = tmp_path / "static_reviews"

    manifest_path = reviews.run_reviews(
        fixture["batch_path"], output_root, workers=1
    )
    manifest = contracts.load_json(manifest_path)
    review_record = manifest["reviews"][0]["review"]
    review = contracts.load_json(output_root / review_record["path"])

    assert manifest["asset_class"] == "static_object"
    assert manifest["route"] == reviews.STATIC_ROUTE
    assert manifest["formal_dataset_registration_authorized"] is False
    assert set(review["raw_pbr_views"]) == set(reviews.VIEW_RECORD_KEYS)
    assert review["clay_geometry"] == {"status": "not_requested"}
    assert review["orientation"]["reference_facing"][
        "canonical_heading_authority"
    ] is False
    assert review["orientation"]["review_orbit"][
        "canonical_heading_authority"
    ] is False
    assert review["orientation"]["canonical_heading"] == {
        "status": "deferred_to_static_finalization",
        "axis": None,
        "derived_from_reference_facing": False,
    }
    assert review["physical_scale"]["status"] == "deferred_to_finalization"
    assert review["physical_scale"]["control_attribute"] is None
    assert review["mesh_readback"]["skins"] == 0
    assert review["mesh_readback"]["animations"] == 0
    with Image.open(output_root / review["contact_sheet"]["path"]) as opened:
        assert opened.size == (960, 640)
    loaded_path, loaded_batch, loaded_reviews = decision_review.load_review_batch(
        manifest_path
    )
    assert loaded_path == manifest_path.resolve()
    assert loaded_batch["review_batch_sha256"] == manifest["review_batch_sha256"]
    assert set(loaded_reviews) == {INSTANCE_ID}

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        reviews.run_reviews(fixture["batch_path"], output_root, workers=1)


def test_optional_clay_geometry_adds_a_separate_five_view_pass(
    tmp_path, monkeypatch
):
    fixture = _static_pixal_fixture(tmp_path)
    _path, _batch, bindings = reviews.load_static_pixal_batch(
        fixture["batch_path"]
    )
    monkeypatch.setattr(reviews.subprocess, "run", _fake_renderer)
    staging = tmp_path / "staging"
    staging.mkdir()

    index = reviews._render_one(
        bindings[INSTANCE_ID],
        staging,
        include_clay_geometry=True,
    )
    review = contracts.load_json(staging / index["review"]["path"])

    assert review["clay_geometry"]["status"] == "included"
    assert set(review["clay_geometry"]["views"]) == set(reviews.VIEW_RECORD_KEYS)
    with Image.open(staging / review["contact_sheet"]["path"]) as opened:
        assert opened.size == (960, 1280)
