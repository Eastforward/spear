#!/usr/bin/env python3
"""Derive a binding pretransform from an immutable manual direction decision.

The review UI defines a canonical generated-animal front of glTF ``+X`` and
allows only yaw about glTF ``+Y``.  Blender's glTF import maps that rotation to
the same signed angle about Blender ``+Z``.  A second cardinal basis rotation
then maps canonical ``+X`` onto the trusted destination rig's longitudinal
axis.  This tool authenticates all inputs and records the exact composition;
it never guesses direction, mirrors geometry, or edits the reviewed GLB.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import run_controlled_animal_lod_binding as direction_contract


SCHEMA = "avengine_controlled_animal_binding_pretransform_v1"
DESTINATION_BASIS_YAW_FROM_POSITIVE_X = {
    "positive-x": 0.0,
    "positive-y": 90.0,
    "negative-x": 180.0,
    "negative-y": -90.0,
}


class PretransformError(RuntimeError):
    """Raised when direction lineage or coordinate composition is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.absolute()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def hash_without(value: Mapping[str, Any], key: str) -> str:
    body = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    return hashlib.sha256(contracts.canonical_json(body).encode("utf-8")).hexdigest()


def normalize_cardinal(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1.0e-9):
        normalized = 180.0
    if math.isclose(normalized, 0.0, abs_tol=1.0e-9):
        normalized = 0.0
    if normalized not in direction_contract.CARDINAL_YAWS:
        raise PretransformError(f"composed binding yaw is not cardinal: {normalized}")
    return normalized


def gltf_yaw_matrix(yaw_deg: float) -> np.ndarray:
    radians = math.radians(float(yaw_deg))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def blender_z_yaw_matrix(yaw_deg: float) -> np.ndarray:
    radians = math.radians(float(yaw_deg))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def derive_pretransform(
    direction_decision_path: Path,
    destination_rig: Path,
    destination_front_axis: str,
) -> dict[str, Any]:
    decision_path = Path(direction_decision_path).absolute()
    if decision_path.is_symlink() or not decision_path.is_file():
        raise PretransformError(f"direction decision is missing: {decision_path}")
    try:
        decision_stub = contracts.load_json(decision_path)
        asset_id = str(decision_stub["asset_id"])
        decision = direction_contract.load_direction_decision(
            decision_path, expected_asset_id=asset_id
        )
    except (KeyError, contracts.ContractError) as error:
        raise PretransformError(f"direction decision authentication failed: {error}") from error

    manual_yaw = float(
        decision["manual_cardinal_yaw_about_gltf_positive_y_deg"]
    )
    recorded_matrix = np.asarray(decision["manual_rotation_matrix_3x3"], dtype=float)
    expected_matrix = gltf_yaw_matrix(manual_yaw)
    if not np.allclose(recorded_matrix, expected_matrix, atol=1.0e-8):
        raise PretransformError("manual rotation matrix does not match its saved yaw")

    if destination_front_axis not in DESTINATION_BASIS_YAW_FROM_POSITIVE_X:
        raise PretransformError(
            f"unsupported destination front axis: {destination_front_axis}"
        )
    rig = Path(destination_rig).absolute()
    if rig.is_symlink() or not rig.is_file() or rig.stat().st_size <= 0:
        raise PretransformError(f"destination rig is missing or unsafe: {rig}")

    # Blender glTF import: (x, y, z)_gltf -> (x, -z, y)_blender.
    gltf_to_blender = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    converted_manual = (
        gltf_to_blender @ expected_matrix @ gltf_to_blender.T
    )
    if not np.allclose(
        converted_manual, blender_z_yaw_matrix(manual_yaw), atol=1.0e-8
    ):
        raise PretransformError("glTF +Y to Blender +Z yaw basis proof failed")

    destination_basis_yaw = DESTINATION_BASIS_YAW_FROM_POSITIVE_X[
        destination_front_axis
    ]
    binding_yaw = normalize_cardinal(manual_yaw + destination_basis_yaw)
    composed = (
        blender_z_yaw_matrix(destination_basis_yaw)
        @ blender_z_yaw_matrix(manual_yaw)
    )
    if not np.allclose(composed, blender_z_yaw_matrix(binding_yaw), atol=1.0e-8):
        raise PretransformError("binding cardinal composition proof failed")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "direction_gate": {
            "decision": file_record(decision_path),
            "decision_sha256": decision["decision_sha256"],
            "status": decision["status"],
            "automatic_orientation_inference_used": False,
            "reviewed_lod": copy.deepcopy(decision["source_prebind_lod"]),
        },
        "destination_rig": {
            **file_record(rig),
            "front_axis_in_blender": destination_front_axis,
        },
        "coordinate_contract": {
            "review_canonical_front_axis": "gltf_positive_x",
            "review_up_axis": "gltf_positive_y",
            "binding_up_axis": "blender_positive_z",
            "gltf_to_blender_matrix_3x3": gltf_to_blender.tolist(),
            "same_signed_yaw_proof_passed": True,
            "manual_cardinal_yaw_about_gltf_positive_y_deg": manual_yaw,
            "destination_basis_yaw_from_blender_positive_x_deg": (
                destination_basis_yaw
            ),
            "composed_target_rotate_z_deg": binding_yaw,
            "composed_rotation_matrix_3x3": composed.tolist(),
            "determinant": float(np.linalg.det(composed)),
            "mirror_used": False,
            "automatic_orientation_inference_used": False,
        },
        "binding_arguments": {
            "target_rotate_z_deg": binding_yaw,
            "semantic_forward_axis": destination_front_axis,
            "flip_x": False,
        },
        "next_gate": "build_fitted_skeleton_then_static_rest_pose_readback",
    }
    payload["manifest_sha256"] = hash_without(payload, "manifest_sha256")
    return payload


def write_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise PretransformError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-decision", type=Path, required=True)
    parser.add_argument("--destination-rig", type=Path, required=True)
    parser.add_argument(
        "--destination-front-axis",
        choices=sorted(DESTINATION_BASIS_YAW_FROM_POSITIVE_X),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = derive_pretransform(
            args.direction_decision,
            args.destination_rig,
            args.destination_front_axis,
        )
        write_no_replace(args.output, payload)
    except (PretransformError, OSError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_BINDING_PRETRANSFORM_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_BINDING_PRETRANSFORM_OK "
        f"asset={payload['asset_id']} "
        f"rotate_z={payload['binding_arguments']['target_rotate_z_deg']} "
        f"output={Path(args.output).absolute()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
