#!/usr/bin/env python3
"""Build authenticated 100k LOD + Walk/Idle runtimes for controlled animals.

This stage consumes one or more immutable controlled ``source_asset_v2``
registries.  It never edits those registries or their Pixal GLBs.  Every job
is written under a new atomic batch root, uses the species-matched Quaternius
Cat/Dog rig, and is read back before the batch is published.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import audit_quadruped_i23d_geometry
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import rocketbox_native_material_canary as immutable
from tools.spike_rlr import runtime_proxy_mesh


BATCH_SCHEMA = "avengine_controlled_animal_lod_binding_batch_v1"
DIRECTION_DECISION_SCHEMA = "controlled_animal_pose_direction_manual_decision_v2"
DIRECTION_DECISION_SCHEMA_V3 = "controlled_animal_pose_direction_manual_decision_v3"
DIRECTION_APPROVED_STATUS = "source_pose_and_cardinal_orientation_approved"
DIRECTION_APPROVED_STATUS_V3 = "source_pose_and_manual_orientation_approved"
CARDINAL_YAWS = {-90.0, 0.0, 90.0, 180.0}
MAX_MANUAL_AXIS_ALIGNMENT_DEG = 45.0
SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
GNU_TIME = Path("/usr/bin/time")
LOD_SCRIPT = SPEAR_ROOT / "tools/blender_create_runtime_proxy_mesh.py"
BIND_SCRIPT = SPEAR_ROOT / "tools/blender_robust_swap_mesh_keep_rig.py"
ANIMATION_TRANSPLANT_SCRIPT = (
    SPEAR_ROOT / "tools/transplant_compatible_glb_animations.py"
)
LATERAL_GAIT_AUDIT_SCRIPT = (
    SPEAR_ROOT / "tools/blender_audit_quadruped_lateral_gait.py"
)
LICENSE_SNAPSHOT = AVENGINE_ROOT / "assets/mesh_library/README.md"
LICENSE_SNAPSHOT_SHA256 = (
    "5887c71ec9a300997bee4445def8f4fb9014ea4e09b36522c1efb9b8eb3a5aef"
)
RIG_SPECS = {
    "cat": {
        "profile_id": "quadruped_cat_v1",
        "skeleton_family": "quaternius_cat",
        "path": AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Cat.glb",
        "sha256": "af2afb5e92c6d9daae98a918f8bd2bcb13ea4d7cfb880020d0d263e4d2f1277e",
    },
    "dog": {
        "profile_id": "quadruped_dog_v1",
        "skeleton_family": "quaternius_dog",
        "path": AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Dog.glb",
        "sha256": "bf9d2fdaf74a36be453edf4516a0b13b042cfce2d2614e0bf3ee24d40d553032",
    },
}
APPROVED_ACTIONS = ["Idle", "Walking"]
LOCKED_PAW_MOTION_PROFILES = {
    "quadruped_dog_locked_paws_v2": {
        "skeleton_family": "quaternius_dog",
        "path": SPEAR_ROOT
        / "tmp/controlled_source_asset_execution_v1/"
        "generated_animal_motion_basis_approved_retarget_foot_ik_v12_post_attachment_20260714/"
        "dog_beagle_three_quarter_seed6102_trellis2/animated_walk_idle.glb",
        "sha256": "083cafc7d99ae1e9e752b512adedef71bf3a124f1d648493874fddc8abc62117",
        "size_bytes": 11_800_824,
        "front_axis": "positive-x",
        "actions": APPROVED_ACTIONS,
        "foot_orientation_policy": "lock_target_rest_world_v1",
        "maximum_lateral_excursion_ratio": 0.005,
        "maximum_terminal_yaw_excursion_degrees": 0.1,
    }
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _verify_file(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed: {path}")


def _matrix_determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1.0e-9):
        return 180.0
    if math.isclose(normalized, 0.0, abs_tol=1.0e-9):
        return 0.0
    return normalized


def _yaw_matrix_y_up(yaw_deg: float) -> list[list[float]]:
    radians = math.radians(float(yaw_deg))
    c, s = math.cos(radians), math.sin(radians)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _decision_binding_yaw(decision: Mapping[str, Any]) -> float:
    if decision.get("schema") == DIRECTION_DECISION_SCHEMA_V3:
        return float(decision["manual_total_yaw_about_gltf_positive_y_deg"])
    return float(decision["manual_cardinal_yaw_about_gltf_positive_y_deg"])


def _locked_paw_motion_spec(profile_id: str | None) -> dict[str, Any] | None:
    if profile_id is None:
        return None
    if profile_id not in LOCKED_PAW_MOTION_PROFILES:
        raise contracts.ContractError(
            f"unknown locked-paw motion profile: {profile_id}"
        )
    spec = copy.deepcopy(LOCKED_PAW_MOTION_PROFILES[profile_id])
    path = Path(spec["path"]).resolve()
    _verify_file(path, spec, label="locked-paw motion carrier")
    spec["profile_id"] = profile_id
    spec["path"] = path
    return spec


def load_direction_decision(
    path: Path, *, expected_asset_id: str
) -> dict[str, Any]:
    """Authenticate the immutable human direction decision used for binding.

    This is deliberately stricter than accepting a caller-provided ``front_axis``:
    the reviewed mesh, exact manual rotation, asset identity, and decision body
    are all content-addressed.  V2 accepts cardinal yaw only.  V3 accepts a
    reviewer-authored small rigid torso-axis alignment plus a cardinal head/tail
    choice; automatic orientation guesses remain forbidden.
    """

    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"direction decision is missing: {path}")
    decision = contracts.load_json(path)
    if not isinstance(decision, dict):
        raise contracts.ContractError("direction decision must be a JSON object")
    schema = decision.get("schema")
    if schema not in {DIRECTION_DECISION_SCHEMA, DIRECTION_DECISION_SCHEMA_V3}:
        raise contracts.ContractError("direction decision schema changed")
    if decision.get("asset_id") != expected_asset_id:
        raise contracts.ContractError("direction decision asset identity changed")
    expected_status = (
        DIRECTION_APPROVED_STATUS_V3
        if schema == DIRECTION_DECISION_SCHEMA_V3
        else DIRECTION_APPROVED_STATUS
    )
    if decision.get("status") != expected_status:
        raise contracts.ContractError("source pose/manual direction is not approved")
    if decision.get("automatic_orientation_inference_used") is not False:
        raise contracts.ContractError(
            "automatic direction inference is forbidden at the manual direction gate"
        )
    if decision.get("decision_sha256") != _hash_without(
        decision, "decision_sha256"
    ):
        raise contracts.ContractError("direction decision hash is invalid")

    if schema == DIRECTION_DECISION_SCHEMA_V3:
        try:
            axis_yaw = float(
                decision["manual_axis_alignment_yaw_about_gltf_positive_y_deg"]
            )
            cardinal_yaw = float(
                decision[
                    "manual_cardinal_head_tail_yaw_about_gltf_positive_y_deg"
                ]
            )
            yaw = float(decision["manual_total_yaw_about_gltf_positive_y_deg"])
        except (KeyError, TypeError, ValueError) as error:
            raise contracts.ContractError(
                "manual two-stage yaw fields are missing"
            ) from error
        if (
            abs(axis_yaw) > MAX_MANUAL_AXIS_ALIGNMENT_DEG
            or cardinal_yaw not in CARDINAL_YAWS
            or not math.isclose(
                yaw, _normalize_yaw(axis_yaw + cardinal_yaw), abs_tol=1.0e-6
            )
        ):
            raise contracts.ContractError("manual two-stage yaw composition is invalid")
        if (
            decision.get("axis_alignment_authority")
            != "human_visual_torso_spine_axis"
            or decision.get("head_tail_authority")
            != "human_visual_head_tail_direction"
        ):
            raise contracts.ContractError("manual two-stage review authority is invalid")
    else:
        try:
            yaw = float(
                decision["manual_cardinal_yaw_about_gltf_positive_y_deg"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise contracts.ContractError("manual cardinal yaw is missing") from error
        if yaw not in CARDINAL_YAWS:
            raise contracts.ContractError(
                f"manual direction must be cardinal, not fine yaw: {yaw}"
            )

    matrix = decision.get("manual_rotation_matrix_3x3")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise contracts.ContractError("manual rotation matrix is invalid")
    try:
        numeric_matrix = [[float(value) for value in row] for row in matrix]
        determinant = _matrix_determinant_3x3(numeric_matrix)
        row_norms = [sum(value * value for value in row) for row in numeric_matrix]
        row_dots = [
            sum(numeric_matrix[i][k] * numeric_matrix[j][k] for k in range(3))
            for i in range(3)
            for j in range(i + 1, 3)
        ]
    except (TypeError, ValueError) as error:
        raise contracts.ContractError(
            "manual rotation matrix is not numeric"
        ) from error
    if (
        abs(determinant - 1.0) > 1.0e-6
        or abs(float(decision.get("determinant", 0.0)) - 1.0) > 1.0e-6
        or any(abs(norm - 1.0) > 1.0e-6 for norm in row_norms)
        or any(abs(dot) > 1.0e-6 for dot in row_dots)
    ):
        raise contracts.ContractError(
            "manual rotation must be a proper orthonormal rotation"
        )
    expected_matrix = _yaw_matrix_y_up(yaw)
    if any(
        abs(numeric_matrix[row][column] - expected_matrix[row][column]) > 1.0e-6
        for row in range(3)
        for column in range(3)
    ):
        raise contracts.ContractError("manual rotation matrix does not match saved yaw")

    reviewed = decision.get("source_prebind_lod")
    if not isinstance(reviewed, dict):
        raise contracts.ContractError("direction decision has no reviewed LOD")
    reviewed_path = Path(str(reviewed.get("absolute_path", ""))).absolute()
    _verify_file(reviewed_path, reviewed, label="direction-reviewed LOD")
    return copy.deepcopy(decision)


def assert_lod_matches_direction_review(
    regenerated_lod: Path, direction_decision: Mapping[str, Any]
) -> None:
    """Prevent a decision for one mesh from authorizing a changed mesh."""

    reviewed = direction_decision.get("source_prebind_lod", {})
    try:
        _verify_file(
            Path(regenerated_lod).absolute(), reviewed, label="reviewed LOD"
        )
    except contracts.ContractError as error:
        raise contracts.ContractError(
            "regenerated LOD does not match the direction-reviewed LOD"
        ) from error


def _rig_spec(source_asset: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy = source_asset.get("taxonomy", {})
    species = taxonomy.get("species")
    if species not in RIG_SPECS:
        raise contracts.ContractError(f"unsupported controlled animal species: {species}")
    spec = RIG_SPECS[species]
    rig = source_asset.get("rig", {})
    if (
        rig.get("profile_id") != spec["profile_id"]
        or rig.get("skeleton_family") != spec["skeleton_family"]
        or rig.get("front_axis") != "positive_x"
        or set(rig.get("actions", [])) != {"Walking", "Idle"}
    ):
        raise contracts.ContractError(
            f"source asset rig contract changed: {source_asset.get('asset_id')}"
        )
    path = Path(spec["path"]).resolve()
    if not path.is_file() or _sha256_file(path) != spec["sha256"]:
        raise contracts.ContractError(f"pinned {species} rig changed: {path}")
    return {**spec, "path": path, "species": species}


def _load_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"source registry is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != source_registry.REGISTRY_SCHEMA
        or payload.get("registry_sha256") != _hash_without(payload, "registry_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("source_asset_count") != len(payload.get("source_assets", []))
    ):
        raise contracts.ContractError(f"source registry contract/hash is invalid: {path}")

    jobs = []
    for index in payload["source_assets"]:
        relative = Path(index.get("source_asset", {}).get("path", ""))
        source_path = (path.parent / relative).resolve()
        try:
            source_path.relative_to(path.parent.resolve())
        except ValueError as error:
            raise contracts.ContractError("source_asset_v2 escaped registry root") from error
        _verify_file(source_path, index["source_asset"], label="source_asset_v2")
        source = contracts.load_json(source_path)
        asset_id = source.get("asset_id")
        if (
            source.get("schema") != contracts.SOURCE_ASSET_SCHEMA
            or source.get("asset_class") != "animal"
            or asset_id != index.get("asset_id")
            or source.get("profile_schema_id") != index.get("profile_schema_id")
            or source.get("request_sha256") != index.get("request_sha256")
            or source.get("sampled_attributes") != index.get("sampled_attributes")
            or source.get("qa", {}).get("static_mesh") != "passed"
            or source.get("qa", {}).get("binding") != "pending"
        ):
            raise contracts.ContractError(f"source_asset_v2 identity changed: {asset_id}")

        raw_record = source.get("artifacts", {}).get("pixal_raw_glb", {})
        if raw_record.get("root_id") != "spear_repo":
            raise contracts.ContractError(f"Pixal GLB root changed: {asset_id}")
        raw_path = (SPEAR_ROOT / raw_record.get("path", "")).resolve()
        try:
            raw_path.relative_to(SPEAR_ROOT.resolve())
        except ValueError as error:
            raise contracts.ContractError("Pixal GLB escaped SPEAR root") from error
        _verify_file(raw_path, raw_record, label="Pixal raw GLB")
        raw_stats = audit_mesh_efficiency.mesh_stats(raw_path)
        if (
            not raw_stats
            or not raw_stats.get("exists")
            or raw_stats.get("triangles", 0) <= 0
            or raw_stats.get("materials", 0) <= 0
            or raw_stats.get("textures", 0) <= 0
            or raw_stats.get("skins") != 0
            or raw_stats.get("animations") != 0
        ):
            raise contracts.ContractError(f"Pixal raw GLB readback failed: {asset_id}")
        jobs.append(
            {
                "asset_id": asset_id,
                "profile_schema_id": source["profile_schema_id"],
                "request_sha256": source["request_sha256"],
                "sampled_attributes": source["sampled_attributes"],
                "target_physical_profile": source["target_physical_profile"],
                "source_asset_path": source_path,
                "source_asset_sha256": index["source_asset"]["sha256"],
                "raw_path": raw_path,
                "raw_record": raw_record,
                "raw_stats": {
                    name: value
                    for name, value in raw_stats.items()
                    if name not in {"path", "exists"}
                },
                "rig": _rig_spec(source),
            }
        )
    return payload, jobs


def load_jobs(
    registry_paths: Sequence[Path],
    asset_ids: Sequence[str] = (),
    direction_decision_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not registry_paths:
        raise contracts.ContractError("at least one source registry is required")
    registries = []
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for path in registry_paths:
        resolved = Path(path).resolve()
        payload, jobs = _load_registry(resolved)
        registries.append(
            {
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "registry_sha256": payload["registry_sha256"],
                "source_asset_count": payload["source_asset_count"],
            }
        )
        for job in jobs:
            if job["asset_id"] in jobs_by_id:
                raise contracts.ContractError(
                    f"duplicate asset across registries: {job['asset_id']}"
                )
            jobs_by_id[job["asset_id"]] = job
    selected = set(asset_ids)
    if selected:
        missing = selected - set(jobs_by_id)
        if missing:
            raise contracts.ContractError(f"requested assets are missing: {sorted(missing)}")
        jobs_by_id = {
            asset_id: job for asset_id, job in jobs_by_id.items() if asset_id in selected
        }
    if not jobs_by_id:
        raise contracts.ContractError("no controlled animal jobs selected")
    if direction_decision_root is None:
        raise contracts.ContractError(
            "a human direction decision root is required before binding"
        )
    decision_root = Path(direction_decision_root).absolute()
    if decision_root.is_symlink() or not decision_root.is_dir():
        raise contracts.ContractError(
            f"direction decision root is missing: {decision_root}"
        )
    for asset_id, job in jobs_by_id.items():
        decision_path = decision_root / f"{asset_id}.json"
        decision = load_direction_decision(
            decision_path, expected_asset_id=asset_id
        )
        job["direction_decision"] = decision
        job["direction_decision_path"] = decision_path
    return registries, [jobs_by_id[key] for key in sorted(jobs_by_id)]


def build_commands(
    job: Mapping[str, Any],
    job_root: Path,
    *,
    target_faces: int,
    bind_output: Path | None = None,
) -> tuple[list[str], list[str]]:
    lod = job_root / "runtime_lod/mesh_runtime_100000_double_sided.glb"
    metadata = job_root / "runtime_lod/mesh_runtime_100000_double_sided.json"
    rigged = (
        Path(bind_output)
        if bind_output is not None
        else job_root / "rigged/animated_100000_double_sided.glb"
    )
    lod_command = [
        str(BLENDER),
        "-b",
        "--python",
        str(LOD_SCRIPT),
        "--",
        "--source",
        str(job["raw_path"]),
        "--output",
        str(lod),
        "--metadata",
        str(metadata),
        "--target-faces",
        str(target_faces),
        "--double-sided",
    ]
    bind_command = [
        str(BLENDER),
        "-b",
        "--python",
        str(BIND_SCRIPT),
        "--",
        "--rig-glb",
        str(job["rig"]["path"]),
        "--new-mesh",
        str(lod),
        "--output",
        str(rigged),
        "--target-rotate-z-deg",
        f"{_decision_binding_yaw(job['direction_decision']):g}",
        "--align-mode",
        "uniform",
        "--weight-mode",
        "region",
        "--segmentation-mode",
        "proximity",
        "--semantic-forward-axis",
        "positive-x",
        "--dampen-head-rotations",
        "0",
        "--dampen-tail-rotations",
        "0",
        "--dampen-foot-rotations",
        "1",
        "--remove-limb-bridges",
        "yes",
        "--delete-limb-bridge-faces",
        "no",
        "--export-action-policy",
        "walk-idle",
    ]
    return lod_command, bind_command


def build_locked_paw_commands(
    target_glb: Path,
    output_glb: Path,
    transplant_manifest: Path,
    lateral_audit: Path,
    spec: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    transplant = [
        str(Path(sys.executable).resolve()),
        str(ANIMATION_TRANSPLANT_SCRIPT),
        "--target-glb",
        str(target_glb),
        "--source-glb",
        str(spec["path"]),
        "--output-glb",
        str(output_glb),
        "--manifest",
        str(transplant_manifest),
    ]
    for action in spec["actions"]:
        transplant.extend(("--action", str(action)))
    lateral = [
        str(BLENDER),
        "-b",
        "--python",
        str(LATERAL_GAIT_AUDIT_SCRIPT),
        "--",
        "--input",
        str(output_glb),
        "--output",
        str(lateral_audit),
        "--front-axis",
        str(spec["front_axis"]),
        "--action",
        "Walking",
        "--samples",
        "41",
    ]
    return transplant, lateral


def validate_locked_paw_audit(
    payload: Mapping[str, Any], output_glb: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("schema") != "avengine_quadruped_lateral_gait_audit_v1"
        or payload.get("input", {}).get("sha256") != _sha256_file(output_glb)
        or payload.get("coordinate_contract", {}).get("front_axis")
        != spec["front_axis"]
        or not (
            str(payload.get("action", "")).lower() == "walking"
            or str(payload.get("action", "")).lower().startswith("walking_")
        )
    ):
        raise contracts.ContractError("locked-paw lateral gait audit identity changed")
    summary = payload.get("summary")
    expected_limbs = {
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    }
    if not isinstance(summary, dict) or set(summary) != expected_limbs:
        raise contracts.ContractError("locked-paw audit limb coverage changed")
    maximum_lateral = max(
        float(record["paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal"])
        for record in summary.values()
    )
    maximum_yaw = max(
        float(record["paw_yaw_excursion_degrees"])
        for record in summary.values()
    )
    if (
        maximum_lateral > float(spec["maximum_lateral_excursion_ratio"])
        or maximum_yaw
        > float(spec["maximum_terminal_yaw_excursion_degrees"])
    ):
        raise contracts.ContractError(
            "locked-paw gait exceeds pinned lateral/yaw thresholds: "
            f"lateral={maximum_lateral:.9g} yaw={maximum_yaw:.9g}"
        )
    return {
        "maximum_lateral_excursion_ratio": maximum_lateral,
        "maximum_terminal_yaw_excursion_degrees": maximum_yaw,
        "lateral_threshold": spec["maximum_lateral_excursion_ratio"],
        "yaw_threshold_degrees": spec[
            "maximum_terminal_yaw_excursion_degrees"
        ],
        "overall": "passed",
    }


def _parse_time_metrics(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    cpu = re.search(r"Percent of CPU this job got: ([0-9.]+)%", text)
    return {
        "max_rss_kib": int(rss.group(1)) if rss else None,
        "cpu_percent": float(cpu.group(1)) if cpu else None,
    }


def _run_timed(command: Sequence[str], log_path: Path, timeout: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("xb") as log:
        result = subprocess.run(
            [str(GNU_TIME), "-v", *command],
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    return {
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        **_parse_time_metrics(log_path),
    }


def _glb_document(path: Path) -> dict[str, Any]:
    return audit_mesh_efficiency._load_glb_json(path)


def validate_lod_and_binding(
    raw_stats: Mapping[str, Any], lod_path: Path, rigged_path: Path, target_faces: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    lod_stats = audit_mesh_efficiency.mesh_stats(lod_path)
    rigged_stats = audit_mesh_efficiency.mesh_stats(rigged_path)
    expected_faces = min(int(raw_stats["triangles"]), int(target_faces))
    if (
        not lod_stats
        or not lod_stats.get("exists")
        or not 0.90 * expected_faces <= lod_stats.get("triangles", 0) <= expected_faces
        or lod_stats.get("materials", 0) <= 0
        or lod_stats.get("textures", 0) <= 0
        or lod_stats.get("skins") != 0
        or lod_stats.get("animations") != 0
    ):
        raise contracts.ContractError("runtime LOD GLB readback failed")
    lod_document = _glb_document(lod_path)
    if not lod_document.get("materials") or not all(
        material.get("doubleSided") is True for material in lod_document["materials"]
    ):
        raise contracts.ContractError("runtime LOD is not double-sided")

    if (
        not rigged_stats
        or not rigged_stats.get("exists")
        or not 0.90 * lod_stats["triangles"]
        <= rigged_stats.get("triangles", 0)
        <= lod_stats["triangles"]
        or rigged_stats.get("materials", 0) <= 0
        or rigged_stats.get("textures", 0) <= 0
        or rigged_stats.get("skins") != 1
        or rigged_stats.get("animations") != 2
    ):
        raise contracts.ContractError("rigged GLB readback failed")
    rigged_document = _glb_document(rigged_path)
    animation_names = [
        animation.get("name") for animation in rigged_document.get("animations", [])
    ]
    channel_paths = {
        channel.get("target", {}).get("path")
        for animation in rigged_document.get("animations", [])
        for channel in animation.get("channels", [])
    }
    if animation_names != APPROVED_ACTIONS or not channel_paths <= {
        "translation",
        "rotation",
    }:
        raise contracts.ContractError(
            f"rigged animation contract changed: {animation_names}/{channel_paths}"
        )
    if not rigged_document.get("materials") or not all(
        material.get("doubleSided") is True
        for material in rigged_document["materials"]
    ):
        raise contracts.ContractError("rigged runtime is not double-sided")
    clean_lod = {
        name: value for name, value in lod_stats.items() if name not in {"path", "exists"}
    }
    clean_rigged = {
        name: value
        for name, value in rigged_stats.items()
        if name not in {"path", "exists"}
    }
    clean_rigged["animation_names"] = animation_names
    clean_rigged["animation_channel_paths"] = sorted(channel_paths)
    return clean_lod, clean_rigged


def _rewrite_runtime_metadata(
    metadata_path: Path, public_runtime_path: Path, *, source_sha256: str
) -> None:
    payload = contracts.load_json(metadata_path)
    physical_runtime = Path(payload.get("runtime_mesh", ""))
    algorithm = payload.get("algorithm")
    if (
        algorithm not in runtime_proxy_mesh.SUPPORTED_RUNTIME_PROXY_ALGORITHMS
        or payload.get("source_mesh_sha256") != source_sha256
        or not physical_runtime.is_file()
        or payload.get("runtime_mesh_sha256") != _sha256_file(physical_runtime)
    ):
        raise contracts.ContractError("runtime LOD metadata contract changed")
    if algorithm == runtime_proxy_mesh.RUNTIME_PROXY_ALGORITHM:
        topology = payload.get("topology", {})
        source_after_weld = topology.get("source_after_position_weld", {})
        runtime_after_decimate = topology.get("runtime_after_decimate", {})
        if (
            not isinstance(topology.get("boundary_cracks_introduced"), int)
            or topology["boundary_cracks_introduced"] > 0
            or not topology.get("position_weld", {}).get("vertices_welded", 0) > 0
            or not isinstance(runtime_after_decimate.get("boundary_edges"), int)
            or runtime_after_decimate["boundary_edges"]
            > source_after_weld.get("boundary_edges", -1)
        ):
            raise contracts.ContractError(
                "welded runtime LOD introduced boundary cracks"
            )
    payload["runtime_mesh"] = str(public_runtime_path.resolve())
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _available_artifacts(paths: Mapping[str, Path], root: Path) -> dict[str, Any]:
    return {
        name: _relative_artifact(path, root)
        for name, path in paths.items()
        if path.is_file() and not path.is_symlink()
    }


def _enforce_prebind_geometry_audit(record: Mapping[str, Any]) -> None:
    decision = record.get("decision", {})
    status = decision.get("status")
    if status == "reject_before_lod_and_binding":
        raise contracts.ContractError(
            "Pixal source geometry rejected before LOD/binding: "
            + ", ".join(decision.get("rejection_reasons", []))
        )
    if status not in {
        "passed_automatic_geometry_measurements",
        "manual_source_geometry_review_required",
    }:
        raise contracts.ContractError("prebind geometry audit status is invalid")


def _direction_gate_record(job: Mapping[str, Any]) -> dict[str, Any]:
    decision = job["direction_decision"]
    schema = decision["schema"]
    record: dict[str, Any] = {
        "schema": schema,
        "status": decision["status"],
        "decision": _relative_artifact(
            Path(job["direction_decision_path"]), SPEAR_ROOT
        ),
        "decision_sha256": decision["decision_sha256"],
        "manual_total_yaw_about_gltf_positive_y_deg": _decision_binding_yaw(
            decision
        ),
        "automatic_orientation_inference_used": False,
        "reviewed_lod": copy.deepcopy(decision["source_prebind_lod"]),
    }
    if schema == DIRECTION_DECISION_SCHEMA_V3:
        record.update(
            manual_axis_alignment_yaw_about_gltf_positive_y_deg=decision[
                "manual_axis_alignment_yaw_about_gltf_positive_y_deg"
            ],
            manual_cardinal_head_tail_yaw_about_gltf_positive_y_deg=decision[
                "manual_cardinal_head_tail_yaw_about_gltf_positive_y_deg"
            ],
            axis_alignment_authority=decision["axis_alignment_authority"],
            head_tail_authority=decision["head_tail_authority"],
        )
    else:
        record["manual_cardinal_yaw_about_gltf_positive_y_deg"] = decision[
            "manual_cardinal_yaw_about_gltf_positive_y_deg"
        ]
    return record


def _run_job(
    job: Mapping[str, Any], staging: Path, public_root: Path, target_faces: int
) -> dict[str, Any]:
    asset_id = str(job["asset_id"])
    job_root = staging / "assets" / asset_id
    paths = {
        "prebind_geometry_audit": job_root / "prebind_geometry_audit.json",
        "lod_glb": job_root / "runtime_lod/mesh_runtime_100000_double_sided.glb",
        "lod_metadata": job_root
        / "runtime_lod/mesh_runtime_100000_double_sided.json",
        "lod_log": job_root / "runtime_lod/blender.log",
        "rigged_prelock_glb": job_root
        / "rigged/animated_100000_double_sided_pre_locked_paws.glb",
        "rigged_glb": job_root / "rigged/animated_100000_double_sided.glb",
        "binding_log": job_root / "rigged/rig.log",
        "animation_transplant_manifest": job_root
        / "rigged/locked_paw_animation_transplant_manifest.json",
        "animation_transplant_log": job_root
        / "rigged/locked_paw_animation_transplant.log",
        "lateral_gait_audit": job_root / "rigged/lateral_gait_audit.json",
        "lateral_gait_audit_log": job_root / "rigged/lateral_gait_audit.log",
    }
    locked_paw = job.get("locked_paw_motion")
    bind_output = paths["rigged_prelock_glb"] if locked_paw else paths["rigged_glb"]
    lod_command, bind_command = build_commands(
        job,
        job_root,
        target_faces=target_faces,
        bind_output=bind_output,
    )
    base = {
        "asset_id": asset_id,
        "profile_schema_id": job["profile_schema_id"],
        "request_sha256": job["request_sha256"],
        "sampled_attributes": job["sampled_attributes"],
        "target_physical_profile": job["target_physical_profile"],
        "source_asset": {
            "path": str(job["source_asset_path"]),
            "sha256": job["source_asset_sha256"],
        },
        "pixal_raw_glb": copy.deepcopy(job["raw_record"]),
        "source_rig": {
            "species": job["rig"]["species"],
            "profile_id": job["rig"]["profile_id"],
            "skeleton_family": job["rig"]["skeleton_family"],
            "root_id": "avengine_repo",
            "path": job["rig"]["path"].relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": job["rig"]["sha256"],
        },
        "raw_mesh_readback": job["raw_stats"],
        "direction_gate": _direction_gate_record(job),
    }
    if locked_paw:
        base["locked_paw_motion"] = {
            "profile_id": locked_paw["profile_id"],
            "skeleton_family": locked_paw["skeleton_family"],
            "source": {
                "path": str(locked_paw["path"]),
                "sha256": locked_paw["sha256"],
                "size_bytes": locked_paw["size_bytes"],
            },
            "foot_orientation_policy": locked_paw["foot_orientation_policy"],
        }
    try:
        paths["prebind_geometry_audit"].parent.mkdir(parents=True, exist_ok=True)
        geometry_record = audit_quadruped_i23d_geometry.audit(
            Path(job["raw_path"]), asset_id
        )
        paths["prebind_geometry_audit"].write_text(
            json.dumps(
                {
                    "schema": audit_quadruped_i23d_geometry.SCHEMA,
                    "created_at": _utc_now(),
                    "purpose": (
                        "prebind_geometry_measurement_without_direction_inference"
                    ),
                    "record": geometry_record,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        base["prebind_geometry_audit"] = copy.deepcopy(geometry_record)
        _enforce_prebind_geometry_audit(geometry_record)
        lod_timing = _run_timed(lod_command, paths["lod_log"], timeout=900)
        if lod_timing["returncode"] != 0:
            return {
                **base,
                "status": "failed_lod",
                "timings": {"lod": lod_timing},
                "artifacts": _available_artifacts(paths, staging),
            }
        public_runtime = (
            public_root
            / "assets"
            / asset_id
            / "runtime_lod/mesh_runtime_100000_double_sided.glb"
        )
        _rewrite_runtime_metadata(
            paths["lod_metadata"],
            public_runtime,
            source_sha256=job["raw_record"]["sha256"],
        )
        assert_lod_matches_direction_review(
            paths["lod_glb"], job["direction_decision"]
        )
        bind_timing = _run_timed(bind_command, paths["binding_log"], timeout=1800)
        if bind_timing["returncode"] != 0:
            return {
                **base,
                "status": "failed_binding",
                "timings": {"lod": lod_timing, "binding": bind_timing},
                "artifacts": _available_artifacts(paths, staging),
            }
        timings = {"lod": lod_timing, "binding": bind_timing}
        if locked_paw:
            _prelock_lod, _prelock_rigged = validate_lod_and_binding(
                job["raw_stats"],
                paths["lod_glb"],
                paths["rigged_prelock_glb"],
                target_faces,
            )
            transplant_command, lateral_command = build_locked_paw_commands(
                paths["rigged_prelock_glb"],
                paths["rigged_glb"],
                paths["animation_transplant_manifest"],
                paths["lateral_gait_audit"],
                locked_paw,
            )
            transplant_timing = _run_timed(
                transplant_command,
                paths["animation_transplant_log"],
                timeout=300,
            )
            timings["locked_paw_animation_transplant"] = transplant_timing
            if transplant_timing["returncode"] != 0:
                return {
                    **base,
                    "status": "failed_locked_paw_animation_transplant",
                    "timings": timings,
                    "artifacts": _available_artifacts(paths, staging),
                }
            transplant_manifest = contracts.load_json(
                paths["animation_transplant_manifest"]
            )
            if (
                transplant_manifest.get("schema")
                != "avengine_compatible_glb_animation_transplant_v1"
                or transplant_manifest.get("target", {}).get("sha256")
                != _sha256_file(paths["rigged_prelock_glb"])
                or transplant_manifest.get("animation_source", {}).get("sha256")
                != locked_paw["sha256"]
                or transplant_manifest.get("output", {}).get("sha256")
                != _sha256_file(paths["rigged_glb"])
                or transplant_manifest.get("transplant", {})
                .get("preservation", {})
                .get("target_binary_prefix_unchanged")
                is not True
            ):
                raise contracts.ContractError(
                    "locked-paw animation transplant manifest changed"
                )
            lateral_timing = _run_timed(
                lateral_command,
                paths["lateral_gait_audit_log"],
                timeout=900,
            )
            timings["locked_paw_lateral_gait_audit"] = lateral_timing
            if lateral_timing["returncode"] != 0:
                return {
                    **base,
                    "status": "failed_locked_paw_lateral_gait_audit",
                    "timings": timings,
                    "artifacts": _available_artifacts(paths, staging),
                }
            base["locked_paw_motion"]["quantitative_audit"] = (
                validate_locked_paw_audit(
                    contracts.load_json(paths["lateral_gait_audit"]),
                    paths["rigged_glb"],
                    locked_paw,
                )
            )
        lod_stats, rigged_stats = validate_lod_and_binding(
            job["raw_stats"], paths["lod_glb"], paths["rigged_glb"], target_faces
        )
        return {
            **base,
            "status": "passed_lod_binding_glb_readback",
            "runtime_lod_readback": lod_stats,
            "rigged_runtime_readback": rigged_stats,
            "timings": timings,
            "artifacts": _available_artifacts(paths, staging),
            "next_gate": "walking_idle_visual_and_contact_qa",
        }
    except (
        contracts.ContractError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        return {
            **base,
            "status": "failed_validation_or_execution",
            "error": str(error),
            "artifacts": _available_artifacts(paths, staging),
        }


def _tool_record(path: Path) -> dict[str, Any]:
    return {
        "root_id": "spear_repo",
        "path": path.relative_to(SPEAR_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _pinned_provenance(
    locked_paw_motion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not LICENSE_SNAPSHOT.is_file()
        or _sha256_file(LICENSE_SNAPSHOT) != LICENSE_SNAPSHOT_SHA256
    ):
        raise contracts.ContractError("Quaternius license snapshot changed")
    rigs = {}
    for species, spec in RIG_SPECS.items():
        path = Path(spec["path"])
        if not path.is_file() or _sha256_file(path) != spec["sha256"]:
            raise contracts.ContractError(f"pinned {species} rig changed")
        rigs[species] = {
            "root_id": "avengine_repo",
            "path": path.relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": spec["sha256"],
            "size_bytes": path.stat().st_size,
            "license": "CC0-1.0",
        }
    result = {
        "license_snapshot": {
            "root_id": "avengine_repo",
            "path": LICENSE_SNAPSHOT.relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": LICENSE_SNAPSHOT_SHA256,
            "size_bytes": LICENSE_SNAPSHOT.stat().st_size,
        },
        "source_rigs": rigs,
        "tools": {
            "prebind_geometry_audit": _tool_record(
                SPEAR_ROOT / "tools/audit_quadruped_i23d_geometry.py"
            ),
            "runtime_lod": _tool_record(LOD_SCRIPT),
            "binding": _tool_record(BIND_SCRIPT),
        },
    }
    if locked_paw_motion:
        result["locked_paw_motion"] = {
            "profile_id": locked_paw_motion["profile_id"],
            "skeleton_family": locked_paw_motion["skeleton_family"],
            "source": {
                "path": str(locked_paw_motion["path"]),
                "sha256": locked_paw_motion["sha256"],
                "size_bytes": locked_paw_motion["size_bytes"],
            },
            "foot_orientation_policy": locked_paw_motion[
                "foot_orientation_policy"
            ],
            "tools": {
                "animation_transplant": _tool_record(
                    ANIMATION_TRANSPLANT_SCRIPT
                ),
                "lateral_gait_audit": _tool_record(
                    LATERAL_GAIT_AUDIT_SCRIPT
                ),
            },
        }
    return result


def run_batch(
    registry_paths: Sequence[Path],
    output_root: Path,
    *,
    workers: int = 8,
    target_faces: int = 100_000,
    asset_ids: Sequence[str] = (),
    direction_decision_root: Path | None = None,
    locked_paw_motion_profile: str | None = None,
) -> Path:
    if not 1 <= workers <= 16:
        raise contracts.ContractError("workers must be between 1 and 16")
    if target_faces != 100_000:
        raise contracts.ContractError("controlled close LOD is pinned to 100000 faces")
    if not BLENDER.is_file() or not GNU_TIME.is_file():
        raise contracts.ContractError("pinned Blender or GNU time is missing")
    locked_paw_motion = _locked_paw_motion_spec(locked_paw_motion_profile)
    provenance = _pinned_provenance(locked_paw_motion)
    registries, jobs = load_jobs(
        registry_paths,
        asset_ids,
        direction_decision_root=direction_decision_root,
    )
    if locked_paw_motion:
        for job in jobs:
            if job["rig"]["skeleton_family"] != locked_paw_motion[
                "skeleton_family"
            ]:
                raise contracts.ContractError(
                    "locked-paw motion profile and selected skeleton family differ"
                )
            job["locked_paw_motion"] = copy.deepcopy(locked_paw_motion)
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    started_at = _utc_now()
    started = time.monotonic()
    try:
        attempts = []
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = {
                executor.submit(
                    _run_job, job, staging, output_root, target_faces
                ): job["asset_id"]
                for job in jobs
            }
            for future in as_completed(futures):
                attempt = future.result()
                attempts.append(attempt)
                print(
                    "CONTROLLED_ANIMAL_LOD_BINDING_JOB_DONE "
                    f"asset={attempt['asset_id']} status={attempt['status']}",
                    flush=True,
                )
        attempts.sort(key=lambda item: item["asset_id"])
        passed = sum(
            item["status"] == "passed_lod_binding_glb_readback" for item in attempts
        )
        failed = len(attempts) - passed
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed" if failed == 0 else "completed_with_failures",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": time.monotonic() - started,
            "source_registries": registries,
            "provenance": provenance,
            "parameters": {
                "target_faces": target_faces,
                "double_sided": True,
                "workers": min(workers, len(jobs)),
                "alignment": "uniform",
                "orientation_source": (
                    "per_asset_authenticated_manual_direction_decision_v2_or_v3"
                ),
                "automatic_orientation_inference": False,
                "manual_binding_yaw": "per_asset_authenticated_decision",
                "flip_x": False,
                "weight_mode": "region",
                "segmentation_mode": "proximity",
                "semantic_forward_axis": "positive-x",
                "prebind_geometry_audit": (
                    audit_quadruped_i23d_geometry.SCHEMA
                ),
                "remove_limb_bridges": True,
                "delete_limb_bridge_faces": False,
                "head_rotation_dampening": 0.0,
                "tail_rotation_dampening": 0.0,
                "foot_rotation_dampening": 1.0,
                "export_actions": APPROVED_ACTIONS,
                "locked_paw_motion_profile": locked_paw_motion_profile,
                "locked_paw_motion_required": locked_paw_motion is not None,
                "ue_animation_channel_paths": ["translation", "rotation"],
            },
            "job_count": len(attempts),
            "passed_count": passed,
            "failed_count": failed,
            "attempts": attempts,
            "automatic_checks": {
                "all_source_registries_reauthenticated": True,
                "all_source_asset_v2_records_reauthenticated": True,
                "all_pixal_raw_glbs_reauthenticated": True,
                "all_direction_decisions_reauthenticated": passed > 0,
                "all_regenerated_lods_match_reviewed_lods": passed > 0,
                "all_successful_sources_passed_prebind_geometry_gate": passed > 0,
                "all_source_rigs_and_license_snapshot_pinned": True,
                "all_successful_lods_glb2_readable_and_double_sided": passed > 0,
                "all_successful_runtimes_have_one_skin": passed > 0,
                "all_successful_runtimes_have_only_idle_and_walking": passed > 0,
                "all_successful_runtimes_have_only_ue_safe_animation_channels": passed
                > 0,
                "all_successful_locked_paw_transplants_preserved_target_geometry": (
                    passed > 0 if locked_paw_motion else None
                ),
                "all_successful_locked_paw_lateral_and_yaw_audits_passed": (
                    passed > 0 if locked_paw_motion else None
                ),
                "visual_animation_and_foot_contact_qa_pending": True,
                "overall": "passed" if failed == 0 else "needs_failure_review",
            },
        }
        manifest["batch_sha256"] = _hash_without(manifest, "batch_sha256")
        contracts.write_json_no_replace(staging / "lod_binding_batch_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("LOD/binding output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "lod_binding_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-faces", type=int, default=100_000)
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument(
        "--direction-decision-root",
        required=True,
        type=Path,
        help=(
            "Directory containing immutable <asset_id>.json manual direction "
            "decisions produced by the direction review UI."
        ),
    )
    parser.add_argument(
        "--locked-paw-motion-profile",
        choices=tuple(sorted(LOCKED_PAW_MOTION_PROFILES)),
        help=(
            "Optional pinned skeleton-family motion carrier. It replaces only "
            "animations after binding and runs lateral/yaw gait QA."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_batch(
            args.registry,
            args.output_root,
            workers=args.workers,
            target_faces=args.target_faces,
            asset_ids=args.asset_id,
            direction_decision_root=args.direction_decision_root,
            locked_paw_motion_profile=args.locked_paw_motion_profile,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_LOD_BINDING_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_LOD_BINDING_OK "
        f"passed={manifest['passed_count']} failed={manifest['failed_count']} "
        f"output={manifest_path}"
    )
    return 0 if manifest["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
