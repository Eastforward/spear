from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools import generated_asset_emitter_contract as contract


HEX_A = "a" * 64
HEX_B = "b" * 64


def test_nonzero_side_coordinate_uses_positive_z_for_anatomical_right():
    assert contract.blender_xyz_to_avengine_local([1.0, -2.0, 3.0]) == [
        1.0,
        3.0,
        2.0,
    ]
    assert contract.avengine_local_to_blender_xyz([1.0, 3.0, 2.0]) == [
        1.0,
        -2.0,
        3.0,
    ]
    contract.validate_coordinate_system(contract.COORDINATE_SYSTEM)
    assert contract.COORDINATE_SYSTEM["id"].endswith("_z_right_m")
    assert contract.COORDINATE_SYSTEM["right_axis"] == [0.0, 0.0, 1.0]


def test_coordinate_contract_rejects_the_historical_left_label():
    changed = copy.deepcopy(contract.COORDINATE_SYSTEM)
    changed["id"] = "avengine_local_x_forward_y_up_z_left_m"

    with pytest.raises(contract.EmitterContractError, match="contract changed"):
        contract.validate_coordinate_system(changed)


def _barycentric_selection():
    return {
        "method": "mesh_surface_barycentric_samples_v1",
        "aggregation": "weighted_centroid",
        "samples": [
            {
                "mesh_name": "mesh",
                "triangle_index": 3,
                "barycentric": [0.2, 0.3, 0.5],
                "weight": 1.0,
            }
        ],
    }


def _bbox_selection():
    return {
        "method": "reviewed_bbox_fraction_nearest_surface_v1",
        "aggregation": "weighted_centroid",
        "maximum_search_distance_fraction": 0.25,
        "samples": [
            {
                "target_fraction_xyz": [0.8, 0.6, 0.7],
                "weight": 1.0,
            }
        ],
    }


@pytest.mark.parametrize("selection", [_barycentric_selection(), _bbox_selection()])
def test_static_selection_contract_accepts_generic_surface_methods(selection):
    contract._validate_selection(selection)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["samples"][0].update(
            {"barycentric": [0.2, 0.3, 0.6]}
        ),
        lambda value: value["samples"][0].update({"triangle_index": -1}),
        lambda value: value["samples"][0].update({"weight": 0.0}),
        lambda value: value["samples"][0].update({"weight": math.inf}),
        lambda value: value.update({"unexpected": True}),
    ],
)
def test_barycentric_selection_fails_closed(mutation):
    selection = _barycentric_selection()
    mutation(selection)

    with pytest.raises(contract.EmitterContractError):
        contract._validate_selection(selection)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["samples"][0].update(
            {"target_fraction_xyz": [0.5, 1.01, 0.5]}
        ),
        lambda value: value.update({"maximum_search_distance_fraction": 0.0}),
        lambda value: value.update({"maximum_search_distance_fraction": math.nan}),
        lambda value: value["samples"][0].update({"weight": -1.0}),
        lambda value: value.update({"unexpected": True}),
    ],
)
def test_bbox_selection_fails_closed(mutation):
    selection = _bbox_selection()
    mutation(selection)

    with pytest.raises(contract.EmitterContractError):
        contract._validate_selection(selection)


def _write_record(path: Path, payload: bytes) -> dict:
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_file_record_rejects_hash_size_and_symlink_tampering(tmp_path):
    source = tmp_path / "source.bin"
    record = _write_record(source, b"authority\n")
    assert contract.validate_file_record(record, label="fixture") == source.resolve()

    changed = copy.deepcopy(record)
    changed["sha256"] = HEX_A
    with pytest.raises(contract.EmitterContractError, match="hash/size changed"):
        contract.validate_file_record(changed, label="fixture")

    changed = copy.deepcopy(record)
    changed["size_bytes"] += 1
    with pytest.raises(contract.EmitterContractError, match="hash/size changed"):
        contract.validate_file_record(changed, label="fixture")

    link = tmp_path / "link.bin"
    link.symlink_to(source)
    linked = copy.deepcopy(record)
    linked["path"] = str(link)
    with pytest.raises(contract.EmitterContractError, match="unsafe"):
        contract.validate_file_record(linked, label="fixture")


def test_canonical_json_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        contract.canonical_json({"value": math.nan})


def test_load_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(contract.EmitterContractError, match="JSON object"):
        contract.load_json_object(path, "fixture")


def _authority_fixture(tmp_path):
    final_glb = tmp_path / "final.glb"
    watertight = tmp_path / "watertight.glb"
    review = tmp_path / "review.png"
    final_record = _write_record(final_glb, b"final-glb\n")
    input_record = _write_record(watertight, b"watertight-glb\n")
    review_record = _write_record(review, b"review-evidence\n")
    finalization_path = tmp_path / "finalization.json"
    finalization = {
        "schema": contract.STATIC_FINALIZATION_SCHEMA,
        "created_at": "2026-07-27T00:00:00+00:00",
        "status": "passed_final_scaled_grounded_canonical_glb",
        "asset_class": "static_object",
        "instance_id": "fixture_static_0001",
        "request_sha256": HEX_A,
        "profile_sha256": HEX_B,
        "input": input_record,
        "output": final_record,
        "coordinate_system": contract.COORDINATE_SYSTEM,
        "heading": {"passed": True, "target_front_axis": "positive-x"},
        "physical_scale": {"passed": True},
        "grounding": {"passed": True},
        "scene_readback": {"mesh_count": 1},
        "formal_dataset_registration_authorized": False,
    }
    finalization_path.write_text(
        json.dumps(finalization, indent=2) + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "anchor.json"
    spec = {
        "schema": contract.STATIC_ANCHOR_SPEC_SCHEMA,
        "instance_id": finalization["instance_id"],
        "request_sha256": finalization["request_sha256"],
        "profile_sha256": finalization["profile_sha256"],
        "finalized_glb_sha256": final_record["sha256"],
        "finalization_manifest_sha256": hashlib.sha256(
            finalization_path.read_bytes()
        ).hexdigest(),
        "anchor_id": "fixture_anchor",
        "anchor_type": "object_speaker",
        "semantic_role": "fixture_feature",
        "selection": _bbox_selection(),
        "review_evidence": review_record,
        "formal_dataset_registration_authorized": False,
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return final_glb, finalization_path, finalization, spec_path, review


def test_finalization_and_anchor_spec_reauthenticate_every_authority(tmp_path):
    (
        final_glb,
        finalization_path,
        _finalization,
        spec_path,
        _review,
    ) = _authority_fixture(tmp_path)

    loaded = contract.validate_static_finalization(finalization_path, final_glb)
    spec = contract.validate_static_anchor_spec(
        spec_path,
        finalization_path=finalization_path,
        input_glb=final_glb,
        finalization=loaded,
    )

    assert spec["anchor_type"] == "object_speaker"


def test_finalization_rejects_output_hash_tampering(tmp_path):
    final_glb, finalization_path, finalization, _spec_path, _review = (
        _authority_fixture(tmp_path)
    )
    finalization["output"]["sha256"] = HEX_B
    finalization_path.write_text(json.dumps(finalization), encoding="utf-8")

    with pytest.raises(contract.EmitterContractError, match="hash/size changed"):
        contract.validate_static_finalization(finalization_path, final_glb)


def test_anchor_spec_rejects_glb_and_review_evidence_tampering(tmp_path):
    final_glb, finalization_path, _finalization, spec_path, review = (
        _authority_fixture(tmp_path)
    )
    loaded = contract.validate_static_finalization(finalization_path, final_glb)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["finalized_glb_sha256"] = "c" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(contract.EmitterContractError, match="GLB hash changed"):
        contract.validate_static_anchor_spec(
            spec_path,
            finalization_path=finalization_path,
            input_glb=final_glb,
            finalization=loaded,
        )

    spec["finalized_glb_sha256"] = hashlib.sha256(
        final_glb.read_bytes()
    ).hexdigest()
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    review.write_bytes(b"changed review evidence\n")
    with pytest.raises(contract.EmitterContractError, match="hash/size changed"):
        contract.validate_static_anchor_spec(
            spec_path,
            finalization_path=finalization_path,
            input_glb=final_glb,
            finalization=loaded,
        )
