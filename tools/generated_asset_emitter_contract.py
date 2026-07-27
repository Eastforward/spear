"""Pure contracts shared by generated-asset emitter measurement tools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MEASUREMENT_SCHEMA = "avengine_asset_emitter_measurement_v2"
STATIC_ANCHOR_SPEC_SCHEMA = "avengine_static_emitter_anchor_spec_v1"
STATIC_FINALIZATION_SCHEMA = "avengine_generated_static_object_finalization_v1"
COORDINATE_SYSTEM = {
    "id": "avengine_local_x_forward_y_up_z_right_m",
    "handedness": "right_handed",
    "forward_axis": [1.0, 0.0, 0.0],
    "up_axis": [0.0, 1.0, 0.0],
    "right_axis": [0.0, 0.0, 1.0],
    "right_rule": "right_equals_forward_cross_up",
}
SHA256_LENGTH = 64


class EmitterContractError(ValueError):
    """Raised when emitter authority or selection evidence is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EmitterContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            not (character.islower() or character.isdigit() or character == "_")
            for character in value
        )
    ):
        raise EmitterContractError(f"{label} must be a lowercase identifier")
    return value


def require_finite_vector(
    value: Any,
    length: int,
    label: str,
) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise EmitterContractError(f"{label} must contain {length} numbers")
    try:
        items = list(value)
    except TypeError as error:
        raise EmitterContractError(
            f"{label} must contain {length} numbers"
        ) from error
    if len(items) != length:
        raise EmitterContractError(f"{label} must contain {length} numbers")
    result = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EmitterContractError(f"{label} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise EmitterContractError(f"{label} must contain only finite numbers")
        result.append(number)
    return result


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    unresolved = Path(path).absolute()
    if (
        unresolved.is_symlink()
        or not unresolved.is_file()
        or unresolved.stat().st_size <= 0
    ):
        raise EmitterContractError(f"missing or unsafe {label}: {unresolved}")
    path = unresolved.resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmitterContractError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise EmitterContractError(f"{label} must be a JSON object")
    return value


def validate_file_record(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise EmitterContractError(f"{label} file record is invalid")
    unresolved = Path(record["path"]).absolute()
    if unresolved.is_symlink():
        raise EmitterContractError(f"missing or unsafe {label}: {unresolved}")
    path = unresolved.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise EmitterContractError(f"{label} path does not match authority")
    if not path.is_file() or path.stat().st_size <= 0:
        raise EmitterContractError(f"missing or unsafe {label}: {path}")
    if (
        isinstance(record["size_bytes"], bool)
        or not isinstance(record["size_bytes"], int)
        or record["size_bytes"] != path.stat().st_size
        or require_sha256(record["sha256"], f"{label} hash") != sha256_file(path)
    ):
        raise EmitterContractError(f"{label} hash/size changed")
    return path


def blender_xyz_to_avengine_local(value: Sequence[float]) -> list[float]:
    """Convert Blender X-forward/Z-up coordinates to glTF X-forward/Y-up/Z-right."""

    x, y, z = require_finite_vector(value, 3, "Blender coordinate")
    return [x, z, -y]


def avengine_local_to_blender_xyz(value: Sequence[float]) -> list[float]:
    """Inverse of :func:`blender_xyz_to_avengine_local`."""

    forward, up, right = require_finite_vector(value, 3, "AVEngine coordinate")
    return [forward, -right, up]


def validate_coordinate_system(value: Any) -> None:
    if value != COORDINATE_SYSTEM:
        raise EmitterContractError("coordinate-system contract changed")
    forward = value["forward_axis"]
    up = value["up_axis"]
    right = value["right_axis"]
    cross = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ]
    if cross != right:
        raise EmitterContractError("coordinate system violates forward cross up = right")


def validate_static_finalization(
    manifest_path: Path,
    input_glb: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_json_object(manifest_path, "static finalization manifest")
    required = {
        "schema",
        "created_at",
        "status",
        "asset_class",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input",
        "output",
        "coordinate_system",
        "heading",
        "physical_scale",
        "grounding",
        "scene_readback",
        "formal_dataset_registration_authorized",
    }
    if set(manifest) != required:
        raise EmitterContractError("static finalization fields are invalid")
    if (
        manifest["schema"] != STATIC_FINALIZATION_SCHEMA
        or manifest["status"] != "passed_final_scaled_grounded_canonical_glb"
        or manifest["asset_class"] != "static_object"
        or manifest["formal_dataset_registration_authorized"] is not False
    ):
        raise EmitterContractError("static finalization contract is not passed")
    require_identifier(manifest["instance_id"], "instance_id")
    require_sha256(manifest["request_sha256"], "request_sha256")
    require_sha256(manifest["profile_sha256"], "profile_sha256")
    validate_coordinate_system(manifest["coordinate_system"])
    output = validate_file_record(
        manifest["output"],
        label="finalized static GLB",
        expected_path=input_glb,
    )
    validate_file_record(manifest["input"], label="watertight static GLB")
    heading = manifest["heading"]
    physical = manifest["physical_scale"]
    grounding = manifest["grounding"]
    if (
        not isinstance(heading, Mapping)
        or heading.get("passed") is not True
        or heading.get("target_front_axis") != "positive-x"
        or not isinstance(physical, Mapping)
        or physical.get("passed") is not True
        or not isinstance(grounding, Mapping)
        or grounding.get("passed") is not True
    ):
        raise EmitterContractError(
            "static heading/physical-scale/grounding gates are not passed"
        )
    if output != input_glb.resolve():
        raise EmitterContractError("finalized static GLB path changed")
    return manifest


def _validate_selection(selection: Any) -> None:
    if not isinstance(selection, Mapping):
        raise EmitterContractError("anchor selection must be an object")
    method = selection.get("method")
    samples = selection.get("samples")
    if selection.get("aggregation") != "weighted_centroid" or not isinstance(
        samples, list
    ) or not samples:
        raise EmitterContractError("anchor selection aggregation/samples are invalid")
    total_weight = 0.0
    if method == "mesh_surface_barycentric_samples_v1":
        if set(selection) != {"method", "samples", "aggregation"}:
            raise EmitterContractError("barycentric anchor selection fields are invalid")
        for sample in samples:
            if not isinstance(sample, Mapping) or set(sample) != {
                "mesh_name",
                "triangle_index",
                "barycentric",
                "weight",
            }:
                raise EmitterContractError("barycentric anchor sample is invalid")
            if not isinstance(sample["mesh_name"], str) or not sample["mesh_name"]:
                raise EmitterContractError("anchor mesh_name is invalid")
            if (
                isinstance(sample["triangle_index"], bool)
                or not isinstance(sample["triangle_index"], int)
                or sample["triangle_index"] < 0
            ):
                raise EmitterContractError("anchor triangle_index is invalid")
            barycentric = require_finite_vector(
                sample["barycentric"], 3, "anchor barycentric"
            )
            if any(value < 0.0 or value > 1.0 for value in barycentric) or not math.isclose(
                sum(barycentric), 1.0, abs_tol=1.0e-6
            ):
                raise EmitterContractError("anchor barycentric weights are invalid")
            weight = require_finite_vector(
                [sample["weight"]], 1, "anchor sample weight"
            )[0]
            if weight <= 0.0:
                raise EmitterContractError("anchor sample weight must be positive")
            total_weight += weight
    elif method == "reviewed_bbox_fraction_nearest_surface_v1":
        allowed = {
            "method",
            "samples",
            "aggregation",
            "maximum_search_distance_fraction",
        }
        if set(selection) != allowed:
            raise EmitterContractError("bbox anchor selection fields are invalid")
        maximum = require_finite_vector(
            [selection["maximum_search_distance_fraction"]],
            1,
            "maximum search distance fraction",
        )[0]
        if not 0.0 < maximum <= 1.0:
            raise EmitterContractError(
                "maximum search distance fraction must be in (0, 1]"
            )
        for sample in samples:
            if not isinstance(sample, Mapping) or set(sample) != {
                "target_fraction_xyz",
                "weight",
            }:
                raise EmitterContractError("bbox anchor sample is invalid")
            fraction = require_finite_vector(
                sample["target_fraction_xyz"], 3, "anchor target fraction"
            )
            if any(value < 0.0 or value > 1.0 for value in fraction):
                raise EmitterContractError(
                    "anchor target fractions must be in [0, 1]"
                )
            weight = require_finite_vector(
                [sample["weight"]], 1, "anchor sample weight"
            )[0]
            if weight <= 0.0:
                raise EmitterContractError("anchor sample weight must be positive")
            total_weight += weight
    else:
        raise EmitterContractError("unsupported static emitter selection method")
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise EmitterContractError("anchor sample weights have zero mass")


def validate_static_anchor_spec(
    spec_path: Path,
    *,
    finalization_path: Path,
    input_glb: Path,
    finalization: Mapping[str, Any],
) -> dict[str, Any]:
    spec = load_json_object(spec_path, "static emitter anchor spec")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "finalized_glb_sha256",
        "finalization_manifest_sha256",
        "anchor_id",
        "anchor_type",
        "semantic_role",
        "selection",
        "review_evidence",
        "formal_dataset_registration_authorized",
    }
    if set(spec) != required:
        raise EmitterContractError("static emitter anchor spec fields are invalid")
    if (
        spec["schema"] != STATIC_ANCHOR_SPEC_SCHEMA
        or spec["anchor_type"] != "object_speaker"
        or spec["formal_dataset_registration_authorized"] is not False
    ):
        raise EmitterContractError("static emitter anchor spec contract is invalid")
    for field in ("instance_id", "request_sha256", "profile_sha256"):
        if spec[field] != finalization[field]:
            raise EmitterContractError(f"anchor spec {field} changed")
    require_identifier(spec["anchor_id"], "anchor_id")
    require_identifier(spec["semantic_role"], "semantic_role")
    if require_sha256(
        spec["finalized_glb_sha256"], "finalized_glb_sha256"
    ) != sha256_file(input_glb):
        raise EmitterContractError("anchor spec finalized GLB hash changed")
    if require_sha256(
        spec["finalization_manifest_sha256"],
        "finalization_manifest_sha256",
    ) != sha256_file(finalization_path):
        raise EmitterContractError("anchor spec finalization manifest hash changed")
    validate_file_record(spec["review_evidence"], label="anchor review evidence")
    _validate_selection(spec["selection"])
    return spec
