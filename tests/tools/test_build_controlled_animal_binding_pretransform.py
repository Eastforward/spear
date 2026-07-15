from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import build_controlled_animal_binding_pretransform as builder
from tools import run_controlled_animal_lod_binding as direction_contract


def _decision(tmp_path: Path, yaw: float = 180.0) -> Path:
    lod = tmp_path / "reviewed.glb"
    lod.write_bytes(b"reviewed glb")
    value = {
        "schema": direction_contract.DIRECTION_DECISION_SCHEMA,
        "manifest_sha256": "a" * 64,
        "asset_id": "horse_fixture",
        "status": direction_contract.DIRECTION_APPROVED_STATUS,
        "automatic_orientation_inference_used": False,
        "manual_cardinal_yaw_about_gltf_positive_y_deg": yaw,
        "manual_rotation_matrix_3x3": builder.gltf_yaw_matrix(yaw).tolist(),
        "determinant": 1.0,
        "source_prebind_lod": {
            "absolute_path": str(lod.absolute()),
            "sha256": builder.sha256_file(lod),
            "size_bytes": lod.stat().st_size,
        },
    }
    value["decision_sha256"] = direction_contract._hash_without(
        value, "decision_sha256"
    )
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_horse_negative_y_rig_composes_reviewed_180_to_blender_positive_90(
    tmp_path: Path,
):
    rig = tmp_path / "horse.glb"
    rig.write_bytes(b"horse rig")

    result = builder.derive_pretransform(
        _decision(tmp_path), rig, "negative-y"
    )

    assert result["binding_arguments"] == {
        "target_rotate_z_deg": 90.0,
        "semantic_forward_axis": "negative-y",
        "flip_x": False,
    }
    contract = result["coordinate_contract"]
    assert contract["manual_cardinal_yaw_about_gltf_positive_y_deg"] == 180
    assert contract["destination_basis_yaw_from_blender_positive_x_deg"] == -90
    assert contract["same_signed_yaw_proof_passed"] is True
    assert contract["mirror_used"] is False
    assert np.isclose(contract["determinant"], 1.0)
    assert result["manifest_sha256"] == builder.hash_without(
        result, "manifest_sha256"
    )


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ("positive-x", 180.0),
        ("positive-y", -90.0),
        ("negative-x", 0.0),
        ("negative-y", 90.0),
    ],
)
def test_all_destination_cardinal_axes_are_deterministic(
    tmp_path: Path, axis: str, expected: float
):
    rig = tmp_path / f"{axis}.glb"
    rig.write_bytes(axis.encode())
    decision = _decision(tmp_path)

    result = builder.derive_pretransform(decision, rig, axis)

    assert result["binding_arguments"]["target_rotate_z_deg"] == expected


def test_mismatched_saved_matrix_is_rejected(tmp_path: Path):
    rig = tmp_path / "horse.glb"
    rig.write_bytes(b"horse rig")
    decision_path = _decision(tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["manual_rotation_matrix_3x3"] = np.eye(3).tolist()
    decision["decision_sha256"] = direction_contract._hash_without(
        decision, "decision_sha256"
    )
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(builder.PretransformError, match="does not match"):
        builder.derive_pretransform(decision_path, rig, "negative-y")


def test_manifest_writer_refuses_to_replace(tmp_path: Path):
    output = tmp_path / "pretransform.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(builder.PretransformError, match="refusing to replace"):
        builder.write_no_replace(output, {"schema": builder.SCHEMA})
