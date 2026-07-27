from __future__ import annotations

import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import generated_asset_emitter_contract as emitter_contract
from tools import review_controlled_static_object_candidates as decisions
from tools import run_controlled_static_object_admission as admission

INSTANCE_ID = "static_fixture_0001"
REQUEST_SHA256 = "1" * 64
PROFILE_SHA256 = "2" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contracts.canonical_json(payload) + "\n", encoding="utf-8")


def _record(path: Path, *, root: Path | None = None) -> dict:
    path = path.resolve()
    return {
        "path": (
            path.relative_to(root.resolve()).as_posix()
            if root is not None
            else str(path)
        ),
        "sha256": admission._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture(autouse=True)
def _mock_full_review_lineage(monkeypatch):
    """The admission fixture is intentionally tiny; production reopens this chain."""

    def load(value):
        decision_batch_path = Path(value["path"])
        batch = contracts.load_json(decision_batch_path)
        reviews = {}
        for index in batch["decisions"]:
            decision = contracts.load_json(
                decision_batch_path.parent / index["record"]["path"]
            )
            reviews[index["instance_id"]] = {
                "path": Path(decision["review"]["path"]),
                "pixal_output_path": Path(decision["pixal_output"]["path"]),
            }
        return Path(value["path"]), {"review_batch_sha256": value["review_batch_sha256"]}, reviews

    monkeypatch.setattr(admission, "_load_review_lineage", load)


def _write_test_glb(path: Path) -> None:
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
        "skins": [],
        "animations": [],
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


def _watertight_parameters() -> dict:
    return {
        "voxel_resolution": 128,
        "target_faces": 10000,
        "smooth_iterations": 1,
        "shrinkwrap_strength": 0.5,
        "post_shrinkwrap_smooth_iterations": 1,
        "torso_fold_repair_iterations": 0,
        "attribute_transfer_backend": "bake",
        "bake_resolution": 512,
        "base_color_encoding_policy": "preserve-bake",
        "base_color_gain": [1.0, 1.0, 1.0],
        "double_sided": False,
    }


def _fixture(tmp_path: Path) -> dict:
    source_root = tmp_path / "source"
    pixal = source_root / "pixal_raw_1024.glb"
    _write_test_glb(pixal)
    decision_root = tmp_path / "decisions"
    decision_path = decision_root / "instances" / INSTANCE_ID / "decision.json"
    review_path = source_root / "static_review.json"
    _write_json(review_path, {"review": "fixture"})
    decision = {
        "schema": decisions.DECISION_SCHEMA,
        "state_classification": "research_candidate",
        "asset_class": admission.STATIC_ASSET_CLASS,
        "route": admission.STATIC_ROUTE,
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "target_physical_profile": {
            "profile_id": "static_fixture_physical_v1",
            "control_attribute": None,
            "selected_value": "fixed",
            "measurement": "height_cm",
            "mode": "absolute_measurement",
            "target_value_cm": 24,
            "tolerance_cm": 2,
        },
        "pixal_output": _record(pixal),
        "review": _record(review_path),
        "decision": decisions.APPROVED,
        "next_gate": "watertight_then_static_finalization",
        "formal_dataset_registration_authorized": False,
    }
    decision["decision_sha256"] = admission._hash_without(
        decision, "decision_sha256"
    )
    _write_json(decision_path, decision)
    decision_index = {
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "pixal_output_sha256": decision["pixal_output"]["sha256"],
        "decision": decisions.APPROVED,
        "decision_sha256": decision["decision_sha256"],
        "record": _record(decision_path, root=decision_root),
    }
    decision_batch_path = decision_root / "decision_batch.json"
    decision_batch = {
        "schema": decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "asset_class": admission.STATIC_ASSET_CLASS,
        "route": admission.STATIC_ROUTE,
        "static_object_review_batch": {
            "path": str(decision_batch_path),
            "sha256": "3" * 64,
            "review_batch_sha256": "4" * 64,
        },
        "decision_count": 1,
        "approved_count": 1,
        "rejected_count": 0,
        "decisions": [decision_index],
        "automatic_checks": {"overall": "passed"},
    }
    decision_batch["decision_batch_sha256"] = admission._hash_without(
        decision_batch, "decision_batch_sha256"
    )
    _write_json(decision_batch_path, decision_batch)

    authority_root = tmp_path / "authorities"
    heading_review = authority_root / "heading_review.png"
    heading_review.parent.mkdir(parents=True, exist_ok=True)
    heading_review.write_bytes(b"reviewed heading\n")
    heading_path = authority_root / "heading.json"
    heading = {
        "schema": "avengine_static_heading_review_v1",
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input_glb_sha256": admission._sha256_file(pixal),
        "review_artifact": _record(heading_review),
        "reviewed_source_front_yaw_deg": 45.0,
        "target_front_axis": "positive-x",
        "decision": "approved_for_positive_x_normalization",
        "formal_dataset_registration_authorized": False,
    }
    _write_json(heading_path, heading)

    anchor_review = authority_root / "anchor_review.png"
    anchor_review.write_bytes(b"reviewed anchor\n")
    anchor_path = authority_root / "anchor.json"
    anchor = {
        "schema": admission.ANCHOR_AUTHORITY_SCHEMA,
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input_glb_sha256": admission._sha256_file(pixal),
        "anchor_id": "speaker",
        "anchor_type": "object_speaker",
        "semantic_role": "primary_sound_source",
        "selection": {
            "method": "reviewed_bbox_fraction_nearest_surface_v1",
            "samples": [
                {"target_fraction_xyz": [0.5, 0.6, 0.5], "weight": 1.0}
            ],
            "aggregation": "weighted_centroid",
            "maximum_search_distance_fraction": 0.5,
        },
        "review_evidence": _record(anchor_review),
        "formal_dataset_registration_authorized": False,
    }
    _write_json(anchor_path, anchor)

    plan = {
        "schema": admission.PLAN_SCHEMA,
        "decision_batch_sha256": decision_batch["decision_batch_sha256"],
        "instances": [
            {
                "instance_id": INSTANCE_ID,
                "heading_evidence_path": str(heading_path.resolve()),
                "anchor_spec_path": str(anchor_path.resolve()),
                "watertight_parameters": _watertight_parameters(),
            }
        ],
        "formal_dataset_registration_authorized": False,
    }
    plan["plan_sha256"] = admission._hash_without(plan, "plan_sha256")
    plan_path = tmp_path / "admission_plan.json"
    _write_json(plan_path, plan)
    return {
        "pixal": pixal,
        "decision": decision_path,
        "decision_batch": decision_batch_path,
        "heading": heading_path,
        "heading_review": heading_review,
        "anchor": anchor_path,
        "anchor_review": anchor_review,
        "plan": plan_path,
        "params": _watertight_parameters(),
    }


def _argument(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1]).resolve()


def _write_watertight_outputs(command: list[str], params: dict) -> None:
    source = _argument(command, "--source")
    output = _argument(command, "--output")
    manifest = _argument(command, "--manifest")
    _write_test_glb(output)
    payload = {
        "schema": "avengine_watertight_textured_runtime_proxy_v1",
        "status": "research_candidate_pending_static_and_animation_qa",
        "input": _record(source),
        "attribute_input": _record(source),
        "output": _record(output),
        "parameters": {**copy.deepcopy(params), "voxel_size": 0.01},
        "topology": {
            "final": {
                "boundary_edges": 0,
                "wire_edges": 0,
                "nonmanifold_edges_over_two_faces": 0,
            }
        },
        "authority_contract": {
            "approved_skeleton_or_animation_touched": False
        },
        "formal_dataset_registration_authorized": False,
    }
    _write_json(manifest, payload)


def _write_finalization_outputs(command: list[str]) -> None:
    input_glb = _argument(command, "--input-glb")
    output = _argument(command, "--output")
    manifest = _argument(command, "--manifest")
    heading_path = _argument(command, "--heading-evidence")
    decision = contracts.load_json(_argument(command, "--static-decision"))
    _write_test_glb(output)
    payload = {
        "schema": emitter_contract.STATIC_FINALIZATION_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "passed_final_scaled_grounded_canonical_glb",
        "asset_class": admission.STATIC_ASSET_CLASS,
        "instance_id": decision["instance_id"],
        "request_sha256": decision["request_sha256"],
        "profile_sha256": decision["profile_sha256"],
        "input": _record(input_glb),
        "output": _record(output),
        "coordinate_system": emitter_contract.COORDINATE_SYSTEM,
        "heading": {
            "passed": True,
            "target_front_axis": "positive-x",
            "evidence": _record(heading_path),
        },
        "physical_scale": {"passed": True},
        "grounding": {"passed": True},
        "scene_readback": {"no_rig_or_animation": True},
        "formal_dataset_registration_authorized": False,
    }
    _write_json(manifest, payload)


def _write_emitter_outputs(command: list[str]) -> None:
    input_glb = _argument(command, "--input-glb")
    finalization_path = _argument(command, "--finalization-manifest")
    anchor_path = _argument(command, "--anchor-spec")
    measurement = _argument(command, "--output")
    marker = _argument(command, "--marker-glb")
    finalization = contracts.load_json(finalization_path)
    anchor = contracts.load_json(anchor_path)
    _write_test_glb(marker)
    payload = {
        "schema": emitter_contract.MEASUREMENT_SCHEMA,
        "created_at": "2026-01-01T00:00:01+00:00",
        "status": "measured_pending_marker_visual_review",
        "asset_class": admission.STATIC_ASSET_CLASS,
        "instance_id": finalization["instance_id"],
        "request_sha256": finalization["request_sha256"],
        "profile_sha256": finalization["profile_sha256"],
        "input": _record(input_glb),
        "finalization_manifest": _record(finalization_path),
        "anchor_spec": _record(anchor_path),
        "coordinate_system": emitter_contract.COORDINATE_SYSTEM,
        "asset_bounds": {
            "minimum_m": [0.0, 0.0, 0.0],
            "maximum_m": [1.0, 1.0, 1.0],
            "extent_m": [1.0, 1.0, 1.0],
        },
        "emitter_anchor": {
            "anchor_id": anchor["anchor_id"],
            "anchor_type": anchor["anchor_type"],
            "semantic_role": anchor["semantic_role"],
            "offset_m": [0.5, 0.6, 0.5],
            "offset_space": "final_scaled_asset_root",
            "method": anchor["selection"]["method"],
            "aggregation": "weighted_centroid",
            "resolved_surface_samples": [],
            "asset_specific_not_class_template": True,
            "animation_required": False,
        },
        "marker_review": {
            "marker_glb": _record(marker),
            "marker_radius_m": 0.01,
            "visual_review": "pending",
        },
        "formal_dataset_registration_authorized": False,
    }
    _write_json(measurement, payload)


def _successful_subprocess(params: dict, calls: list[str]):
    def run(command, *, cwd, stdout, stderr, timeout, check):
        command = list(command)
        script = _argument(command, "--python").name
        calls.append(script)
        stdout.write(f"fake {script} passed\n".encode())
        if script == admission.WATERTIGHT_TOOL.name:
            _write_watertight_outputs(command, params)
        elif script == admission.FINALIZER_TOOL.name:
            _write_finalization_outputs(command)
        elif script == admission.EMITTER_TOOL.name:
            _write_emitter_outputs(command)
        else:
            raise AssertionError(f"unexpected admission tool: {script}")
        return subprocess.CompletedProcess(command, 0)

    return run


def _prepare_execution(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    fixture = _fixture(tmp_path)
    fake_blender = tmp_path / "blender"
    fake_blender.write_bytes(b"fixture executable\n")
    monkeypatch.setattr(admission, "BLENDER", fake_blender)
    monkeypatch.setattr(admission.immutable, "_seal_readonly_tree", lambda _root: None)
    return fixture, tmp_path / "published"


def test_validate_only_reauthenticates_frozen_authorities_and_detects_tamper(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    assert (
        admission.run_admission(
            fixture["decision_batch"],
            fixture["plan"],
            tmp_path / "unused",
            validate_only=True,
        )
        is None
    )
    fixture["heading_review"].write_bytes(b"tampered heading review\n")
    with pytest.raises(admission.AdmissionError, match="hash/size changed"):
        admission.run_admission(
            fixture["decision_batch"],
            fixture["plan"],
            tmp_path / "still_unused",
            validate_only=True,
        )


def test_commands_are_data_driven_and_do_not_contain_object_name_branches(tmp_path):
    fixture = _fixture(tmp_path)
    inputs = admission.validate_admission_inputs(
        fixture["decision_batch"], fixture["plan"]
    )
    commands, _paths = admission.build_stage_commands(
        inputs["jobs"][INSTANCE_ID], tmp_path / "stage"
    )
    watertight = commands["watertight"]
    assert _argument(watertight, "--source") == fixture["pixal"].resolve()
    assert watertight[watertight.index("--voxel-resolution") + 1] == "128"
    assert watertight[watertight.index("--base-color-gain") + 1 :][-3:] == [
        "1.0",
        "1.0",
        "1.0",
    ]
    source = Path(admission.__file__).read_text(encoding="utf-8").lower()
    for object_name in ("telephone", "doorbell", "kettle", "microwave", "alarm_clock"):
        assert object_name not in source


def test_partial_failure_preserves_logs_and_stops_before_later_stage(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    calls = []

    def fail_finalization(command, *, cwd, stdout, stderr, timeout, check):
        command = list(command)
        script = _argument(command, "--python").name
        calls.append(script)
        stdout.write(f"fake {script}\n".encode())
        if script == admission.WATERTIGHT_TOOL.name:
            _write_watertight_outputs(command, fixture["params"])
            return subprocess.CompletedProcess(command, 0)
        if script == admission.FINALIZER_TOOL.name:
            stdout.write(b"forced finalization failure\n")
            return subprocess.CompletedProcess(command, 17)
        raise AssertionError("emitter stage must not run after finalization failure")

    monkeypatch.setattr(admission.subprocess, "run", fail_finalization)
    with pytest.raises(admission.AdmissionError, match="failure_evidence="):
        admission.run_admission(
            fixture["decision_batch"], fixture["plan"], output_root
        )
    assert calls == [
        admission.WATERTIGHT_TOOL.name,
        admission.FINALIZER_TOOL.name,
    ]
    assert not output_root.exists()
    failures = list(tmp_path.glob(f"{output_root.name}.failed_*"))
    assert len(failures) == 1
    failure = contracts.load_json(
        failures[0] / "admission_failure_manifest.json"
    )
    assert failure["failed_stage"] == "finalization"
    assert [item["returncode"] for item in failure["executed_commands"]] == [0, 17]
    assert failure["formal_dataset_registration_authorized"] is False
    assert (
        failures[0]
        / "instances"
        / INSTANCE_ID
        / "02_finalization"
        / "blender.log"
    ).read_bytes().endswith(b"forced finalization failure\n")
    assert not (
        failures[0]
        / "instances"
        / INSTANCE_ID
        / "03_emitter"
        / "emitter_measurement.json"
    ).exists()


def test_midrun_external_authority_tamper_cannot_change_frozen_execution(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    calls = []

    def mutate_heading(command, *, cwd, stdout, stderr, timeout, check):
        command = list(command)
        script = _argument(command, "--python").name
        calls.append(script)
        if script == admission.WATERTIGHT_TOOL.name:
            _write_watertight_outputs(command, fixture["params"])
            heading = contracts.load_json(fixture["heading"])
            heading["reviewed_source_front_yaw_deg"] = -45.0
            _write_json(fixture["heading"], heading)
        elif script == admission.FINALIZER_TOOL.name:
            _write_finalization_outputs(command)
        elif script == admission.EMITTER_TOOL.name:
            _write_emitter_outputs(command)
        else:
            raise AssertionError(script)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admission.subprocess, "run", mutate_heading)
    admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert calls == [admission.WATERTIGHT_TOOL.name, admission.FINALIZER_TOOL.name, admission.EMITTER_TOOL.name]
    bound = contracts.load_json(output_root / "instances" / INSTANCE_ID / "02_finalization" / "bound_heading_evidence.json")
    assert bound["reviewed_source_front_yaw_deg"] == 45.0


def test_success_rebinds_only_future_hashes_publishes_receipts_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        _successful_subprocess(fixture["params"], calls),
    )
    manifest_path = admission.run_admission(
        fixture["decision_batch"], fixture["plan"], output_root
    )
    assert calls == [
        admission.WATERTIGHT_TOOL.name,
        admission.FINALIZER_TOOL.name,
        admission.EMITTER_TOOL.name,
    ]
    batch = contracts.load_json(manifest_path)
    assert batch["status"] == (
        "passed_all_instances_pending_emitter_marker_review"
    )
    assert batch["marker_review"]["status"] == "pending"
    assert batch["formal_dataset_registration_authorized"] is False
    assert batch["batch_sha256"] == admission._hash_without(batch, "batch_sha256")

    instance_root = output_root / "instances" / INSTANCE_ID
    source_heading = contracts.load_json(fixture["heading"])
    bound_heading_path = (
        instance_root / "02_finalization" / "bound_heading_evidence.json"
    )
    bound_heading = contracts.load_json(bound_heading_path)
    assert {
        key: value
        for key, value in bound_heading.items()
            if key not in {"input_glb_sha256", "review_artifact"}
    } == {
        key: value
        for key, value in source_heading.items()
            if key not in {"input_glb_sha256", "review_artifact"}
    }
    assert bound_heading["review_artifact"]["sha256"] == source_heading["review_artifact"]["sha256"]
    assert Path(bound_heading["review_artifact"]["path"]).is_relative_to(output_root)
    watertight_glb = instance_root / "01_watertight" / "watertight.glb"
    assert bound_heading["input_glb_sha256"] == admission._sha256_file(
        watertight_glb
    )

    source_anchor = contracts.load_json(fixture["anchor"])
    bound_anchor_path = instance_root / "03_emitter" / "bound_anchor_spec.json"
    bound_anchor = contracts.load_json(bound_anchor_path)
    assert bound_anchor["schema"] == emitter_contract.STATIC_ANCHOR_SPEC_SCHEMA
    for key, value in source_anchor.items():
        if key not in {"schema", "input_glb_sha256", "review_evidence"}:
            assert bound_anchor[key] == value
    assert bound_anchor["review_evidence"]["sha256"] == source_anchor["review_evidence"]["sha256"]
    assert Path(bound_anchor["review_evidence"]["path"]).is_relative_to(output_root)
    final_glb = instance_root / "02_finalization" / "finalized.glb"
    final_manifest = (
        instance_root / "02_finalization" / "finalization_manifest.json"
    )
    assert bound_anchor["finalized_glb_sha256"] == admission._sha256_file(
        final_glb
    )
    assert bound_anchor["finalization_manifest_sha256"] == (
        admission._sha256_file(final_manifest)
    )
    emitter_contract.validate_static_anchor_spec(
        bound_anchor_path,
        finalization_path=final_manifest,
        input_glb=final_glb,
        finalization=contracts.load_json(final_manifest),
    )

    watertight_manifest = contracts.load_json(
        instance_root / "01_watertight" / "watertight_manifest.json"
    )
    emitter = contracts.load_json(
        instance_root / "03_emitter" / "emitter_measurement.json"
    )
    assert Path(watertight_manifest["output"]["path"]).is_relative_to(output_root)
    assert Path(emitter["input"]["path"]).is_relative_to(output_root)
    assert emitter["marker_review"]["visual_review"] == "pending"
    receipt = contracts.load_json(instance_root / "admission_receipt.json")
    assert receipt["marker_review"] == "pending"
    assert receipt["formal_dataset_registration_authorized"] is False
    assert receipt["receipt_sha256"] == admission._hash_without(
        receipt, "receipt_sha256"
    )
    assert set(receipt["stage_receipts"]) == {
        "watertight",
        "finalization",
        "emitter_measurement",
    }
    expected_dependencies = {
        "watertight": set(),
        "finalization": {"generated_asset_emitter_contract.py"},
        "emitter_measurement": {"generated_asset_emitter_contract.py"},
    }
    for stage_name, stage_record in receipt["stage_receipts"].items():
        stage_receipt = contracts.load_json(
            output_root / stage_record["path"]
        )
        command_manifest_record = stage_receipt["command_input_manifest"]
        assert Path(command_manifest_record["path"]).is_relative_to(output_root)
        command_manifest = contracts.load_json(
            Path(command_manifest_record["path"])
        )
        assert command_manifest["schema"] == admission.COMMAND_INPUT_MANIFEST_SCHEMA
        assert command_manifest["command"] == stage_receipt["command"]
        assert command_manifest["command_sha256"] == admission._json_sha256(
            stage_receipt["command"]
        )
        assert command_manifest["manifest_sha256"] == admission._hash_without(
            command_manifest, "manifest_sha256"
        )
        assert Path(command_manifest["python_tool"]["path"]).is_relative_to(
            output_root
        )
        assert set(command_manifest["python_dependencies"]) == expected_dependencies[
            stage_name
        ]

    admission._validate_published_batch(output_root)

    calls_before = list(calls)
    with pytest.raises(admission.AdmissionError, match="refusing to replace"):
        admission.run_admission(
            fixture["decision_batch"], fixture["plan"], output_root
        )
    assert calls == calls_before


def test_anchor_authority_must_bind_approved_raw_pixal_hash(tmp_path):
    fixture = _fixture(tmp_path)
    anchor = contracts.load_json(fixture["anchor"])
    anchor["input_glb_sha256"] = "0" * 64
    _write_json(fixture["anchor"], anchor)
    with pytest.raises(admission.AdmissionError, match="bind the approved raw Pixal"):
        admission.validate_admission_inputs(fixture["decision_batch"], fixture["plan"])


def test_emitter_metadata_mismatch_fails_closed(tmp_path, monkeypatch):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)

    def wrong_emitter(command, *, cwd, stdout, stderr, timeout, check):
        command = list(command)
        script = _argument(command, "--python").name
        if script == admission.WATERTIGHT_TOOL.name:
            _write_watertight_outputs(command, fixture["params"])
        elif script == admission.FINALIZER_TOOL.name:
            _write_finalization_outputs(command)
        else:
            _write_emitter_outputs(command)
            measurement = _argument(command, "--output")
            payload = contracts.load_json(measurement)
            payload["emitter_anchor"]["semantic_role"] = "wrong_role"
            _write_json(measurement, payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admission.subprocess, "run", wrong_emitter)
    with pytest.raises(admission.AdmissionError, match="semantic_role differs"):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert not output_root.exists()


def test_receipts_use_frozen_source_records_after_external_mutation(tmp_path, monkeypatch):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(admission.subprocess, "run", _successful_subprocess(fixture["params"], []))
    original = admission._publish_job_receipts

    def mutate_after_snapshot(result, staging, public_root):
        payload = contracts.load_json(fixture["decision"])
        payload["next_gate"] = "tampered_after_freeze"
        payload["decision_sha256"] = admission._hash_without(payload, "decision_sha256")
        _write_json(fixture["decision"], payload)
        return original(result, staging, public_root)

    monkeypatch.setattr(admission, "_publish_job_receipts", mutate_after_snapshot)
    admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    receipt = contracts.load_json(output_root / "instances" / INSTANCE_ID / "admission_receipt.json")
    batch = contracts.load_json(fixture["decision_batch"])
    assert receipt["decision"]["sha256"] == batch["decisions"][0]["record"]["sha256"]


def test_rename_noreplace_refuses_concurrent_empty_directory(tmp_path):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "new").write_text("new", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "peer").write_text("peer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        admission._rename_noreplace(source, destination)
    assert (destination / "peer").read_text(encoding="utf-8") == "peer"
    assert (source / "new").read_text(encoding="utf-8") == "new"


def test_staged_validation_failure_never_publishes_root(tmp_path, monkeypatch):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(admission.subprocess, "run", _successful_subprocess(fixture["params"], []))
    monkeypatch.setattr(
        admission,
        "_validate_staged_rebase",
        lambda *_args: (_ for _ in ()).throw(admission.AdmissionError("staged reject")),
    )
    with pytest.raises(admission.AdmissionError, match="failure_evidence="):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert not output_root.exists()
    assert len(list(tmp_path.glob(f"{output_root.name}.failed_*"))) == 1


def test_resealed_command_input_chain_cannot_hide_frozen_tool_tamper(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        _successful_subprocess(fixture["params"], []),
    )
    original = admission._validate_staged_rebase

    def tamper_and_reseal(result, staging, public_root):
        paths = result["paths"]
        frozen_tool = Path(
            result["command_inputs"]["tools"]["watertight"]["frozen_path"]
        )
        frozen_tool.chmod(0o644)
        frozen_tool.write_bytes(b"re-signed but substituted watertight tool\n")

        command_manifest_path = paths["watertight_command_inputs"]
        command_manifest = contracts.load_json(command_manifest_path)
        command_manifest["python_tool"] = admission._file_record(
            frozen_tool,
            recorded_path=public_root / frozen_tool.relative_to(staging),
        )
        command_manifest["manifest_sha256"] = admission._hash_without(
            command_manifest, "manifest_sha256"
        )
        admission._replace_json(command_manifest_path, command_manifest)

        stage_receipt_path = paths["watertight_dir"] / "stage_receipt.json"
        stage_receipt = contracts.load_json(stage_receipt_path)
        stage_receipt["command_input_manifest"] = admission._file_record(
            command_manifest_path,
            recorded_path=public_root / command_manifest_path.relative_to(staging),
        )
        stage_receipt["receipt_sha256"] = admission._hash_without(
            stage_receipt, "receipt_sha256"
        )
        admission._replace_json(stage_receipt_path, stage_receipt)

        job_receipt_path = paths["job_receipt"]
        job_receipt = contracts.load_json(job_receipt_path)
        job_receipt["stage_receipts"]["watertight"] = _record(
            stage_receipt_path, root=staging
        )
        job_receipt["receipt_sha256"] = admission._hash_without(
            job_receipt, "receipt_sha256"
        )
        admission._replace_json(job_receipt_path, job_receipt)

        batch_path = staging / "static_object_admission_batch_manifest.json"
        batch = contracts.load_json(batch_path)
        batch["jobs"][0]["job_receipt"] = _record(job_receipt_path, root=staging)
        batch["batch_sha256"] = admission._hash_without(batch, "batch_sha256")
        admission._replace_json(batch_path, batch)
        return original(result, staging, public_root)

    monkeypatch.setattr(admission, "_validate_staged_rebase", tamper_and_reseal)
    with pytest.raises(admission.AdmissionError, match="frozen watertight tool changed"):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert not output_root.exists()


def test_dangling_output_root_symlink_is_rejected_before_resolution(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    output_root.symlink_to(tmp_path / "redirected_publication_root")
    with pytest.raises(admission.AdmissionError, match="must not be a symlink"):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert not (tmp_path / "redirected_publication_root").exists()


def test_post_seal_publication_collision_preserves_failure_in_fresh_sibling(
    tmp_path, monkeypatch
):
    original_sealer = admission.immutable._seal_readonly_tree
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    # This regression specifically needs the production readonly seal, rather
    # than the fixture's usual no-op, before publication races with a peer.
    monkeypatch.setattr(admission.immutable, "_seal_readonly_tree", original_sealer)
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        _successful_subprocess(fixture["params"], []),
    )
    original_rename = admission._rename_noreplace
    created_peer = False

    def collide_on_publication(source, destination):
        nonlocal created_peer
        if destination == output_root and not created_peer:
            created_peer = True
            output_root.mkdir()
            (output_root / "peer").write_text("peer", encoding="utf-8")
        return original_rename(source, destination)

    monkeypatch.setattr(admission, "_rename_noreplace", collide_on_publication)
    with pytest.raises(admission.AdmissionError, match="failure_evidence="):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert (output_root / "peer").read_text(encoding="utf-8") == "peer"
    failures = list(tmp_path.glob(f"{output_root.name}.failed_*"))
    assert len(failures) == 1
    failure = contracts.load_json(failures[0] / "admission_failure_manifest.json")
    assert failure["error_type"] == "FileExistsError"
    assert (
        failures[0]
        / "instances"
        / INSTANCE_ID
        / "03_emitter"
        / "blender.log"
    ).is_file()


def test_midrun_tool_source_swap_fails_closed_after_using_frozen_copy(
    tmp_path, monkeypatch
):
    fixture, output_root = _prepare_execution(tmp_path, monkeypatch)
    tool_root = tmp_path / "mutable_tool_sources"
    tool_root.mkdir()
    watertight = tool_root / admission.WATERTIGHT_TOOL.name
    finalizer = tool_root / admission.FINALIZER_TOOL.name
    emitter = tool_root / admission.EMITTER_TOOL.name
    for path in (watertight, finalizer, emitter):
        path.write_text("original tool bytes\n", encoding="utf-8")
    monkeypatch.setattr(admission, "WATERTIGHT_TOOL", watertight)
    monkeypatch.setattr(admission, "FINALIZER_TOOL", finalizer)
    monkeypatch.setattr(admission, "EMITTER_TOOL", emitter)
    calls = []
    invoked_tools = []
    delegate = _successful_subprocess(fixture["params"], calls)

    def swap_after_watertight(command, *, cwd, stdout, stderr, timeout, check):
        invoked_tools.append(_argument(list(command), "--python"))
        result = delegate(
            command,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=check,
        )
        if _argument(list(command), "--python").name == watertight.name:
            watertight.write_text("swapped tool bytes\n", encoding="utf-8")
        return result

    monkeypatch.setattr(admission.subprocess, "run", swap_after_watertight)
    with pytest.raises(admission.AdmissionError, match="watertight tool changed"):
        admission.run_admission(fixture["decision_batch"], fixture["plan"], output_root)
    assert calls == [watertight.name]
    assert invoked_tools[0].parent.name == "tools"
    assert invoked_tools[0].parents[1].name == ".runtime_commands"
    assert not output_root.exists()
