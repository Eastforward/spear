#!/usr/bin/env python3
"""Shared canonical arm/leg motion-basis review contracts."""

from __future__ import annotations

import math
import datetime
import hashlib
import json
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ALLOWED_YAW_DEGREES = (0, -90, 90, 180)
CANONICAL_FRONT = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
CANONICAL_UP = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
LIMB_CHAINS = {
    "left_arm": ("left_upper_arm", "left_forearm", "left_hand"),
    "right_arm": ("right_upper_arm", "right_forearm", "right_hand"),
    "left_leg": ("left_thigh", "left_calf", "left_foot"),
    "right_leg": ("right_thigh", "right_calf", "right_foot"),
}
_BODY_ROLES = ("left_clavicle", "right_clavicle", "left_thigh", "right_thigh")
CANDIDATE_ANGLES = {
    "yaw_000": 0,
    "yaw_m090": -90,
    "yaw_p090": 90,
    "yaw_180": 180,
}
VIEWS = ("front", "side", "top", "feet", "skeleton")
BUNDLE_SCHEMA = "shared_limb_motion_basis_review_v1"
SELECTION_SCHEMA = "retarget_motion_basis_correction_v1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
AXIAL_MAXIMUM_ABS_DEGREES = {
    "torso_lateral_tilt_deg": 2.0,
    "shoulder_roll_deg": 2.0,
    "hip_roll_deg": 3.0,
    "head_body_lateral_tilt_deg": 2.0,
    "neck_head_lateral_tilt_deg": 3.0,
    "head_bone_lateral_tilt_deg": 3.0,
}


class MotionBasisReviewError(RuntimeError):
    """A shared motion-basis review invariant failed."""


def yaw_matrix(degrees: int) -> np.ndarray:
    """Return one of the explicitly reviewed proper rotations around ``UP +Z``."""

    if isinstance(degrees, bool) or not isinstance(degrees, int):
        raise MotionBasisReviewError("yaw must be one of the allowed integer angles")
    if degrees not in ALLOWED_YAW_DEGREES:
        raise MotionBasisReviewError(
            f"yaw must be one of the allowed angles: {ALLOWED_YAW_DEGREES}"
        )
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    value = np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    if (
        not np.isfinite(value).all()
        or not np.allclose(value.T @ value, np.eye(3), atol=1.0e-12)
        or not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=1.0e-12)
        or not np.allclose(value @ CANONICAL_UP, CANONICAL_UP, atol=1.0e-12)
    ):
        raise MotionBasisReviewError("allowed yaw did not produce a proper UP +Z rotation")
    return value


def _point(frame: Mapping[str, Sequence[float]], role: str, frame_index: int) -> np.ndarray:
    if role not in frame:
        raise MotionBasisReviewError(
            f"four-limb frame {frame_index} is missing semantic role {role}"
        )
    value = np.asarray(frame[role], dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise MotionBasisReviewError(
            f"four-limb frame {frame_index} semantic role {role} is not finite 3D"
        )
    return value


def _horizontal_normalized(value: np.ndarray, description: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[2] = 0.0
    length = float(np.linalg.norm(result))
    if not math.isfinite(length) or length <= 1.0e-10:
        raise MotionBasisReviewError(f"{description} is horizontally degenerate")
    return result / length


def _body_basis(
    frame: Mapping[str, Sequence[float]], frame_index: int
) -> tuple[np.ndarray, np.ndarray]:
    points = {role: _point(frame, role, frame_index) for role in _BODY_ROLES}
    body_right = _horizontal_normalized(
        (points["right_clavicle"] - points["left_clavicle"])
        + (points["right_thigh"] - points["left_thigh"]),
        f"four-limb frame {frame_index} body lateral basis",
    )
    body_forward = _horizontal_normalized(
        np.cross(body_right, CANONICAL_UP),
        f"four-limb frame {frame_index} body forward basis",
    )
    if float(np.dot(body_forward, CANONICAL_FRONT)) < 0.0:
        body_forward *= -1.0
    return body_right, body_forward


def compute_four_limb_motion_metrics(
    frames: Sequence[Mapping[str, Sequence[float]]], *, fps: int
) -> dict[str, Any]:
    """Measure arm and leg swing/bend planes in the same signed body basis."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise MotionBasisReviewError("four-limb FPS must be a positive integer")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise MotionBasisReviewError("four-limb frames must be a sequence")
    if len(frames) < 2:
        raise MotionBasisReviewError("four-limb review needs at least two frames")

    samples: dict[str, list[dict[str, float | None]]] = {
        limb: [] for limb in LIMB_CHAINS
    }
    frame_evidence: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames, start=1):
        if not isinstance(frame, Mapping):
            raise MotionBasisReviewError(
                f"four-limb frame {frame_index} must be a semantic mapping"
            )
        body_right, body_forward = _body_basis(frame, frame_index)
        limb_evidence = {}
        for limb, (anchor_role, middle_role, endpoint_role) in LIMB_CHAINS.items():
            anchor = _point(frame, anchor_role, frame_index)
            middle = _point(frame, middle_role, frame_index)
            endpoint = _point(frame, endpoint_role, frame_index)
            anchor_to_middle = middle - anchor
            middle_to_endpoint = endpoint - middle
            plane_normal_raw = np.cross(anchor_to_middle, middle_to_endpoint)
            plane_normal_length = float(np.linalg.norm(plane_normal_raw))
            chord_length = float(np.linalg.norm(endpoint - anchor))
            bend_m = plane_normal_length / max(chord_length, 1.0e-12)
            if plane_normal_length <= 1.0e-10:
                normal_lateral = None
                normal_forward = None
            else:
                plane_normal = plane_normal_raw / plane_normal_length
                normal_lateral = abs(float(np.dot(plane_normal, body_right)))
                normal_forward = abs(float(np.dot(plane_normal, body_forward)))
            relative = endpoint - anchor
            sample = {
                "endpoint_forward_m": float(np.dot(relative, body_forward)),
                "endpoint_lateral_m": float(np.dot(relative, body_right)),
                "bend_m": bend_m,
                "plane_normal_dot_lateral_abs": normal_lateral,
                "plane_normal_dot_forward_abs": normal_forward,
            }
            samples[limb].append(sample)
            limb_evidence[limb] = sample
        frame_evidence.append(
            {
                "frame": frame_index,
                "body_forward": body_forward.tolist(),
                "body_right": body_right.tolist(),
                "limbs": limb_evidence,
            }
        )

    summaries: dict[str, dict[str, Any]] = {}
    classifications = []
    for limb, values in samples.items():
        forward = [float(value["endpoint_forward_m"]) for value in values]
        lateral = [float(value["endpoint_lateral_m"]) for value in values]
        valid = [
            value
            for value in values
            if value["plane_normal_dot_lateral_abs"] is not None
            and float(value["bend_m"]) >= 0.005
        ]
        if not valid:
            raise MotionBasisReviewError(
                f"{limb} has no measurable bend plane across the reviewed frames"
            )
        forward_excursion = max(forward) - min(forward)
        lateral_excursion = max(lateral) - min(lateral)
        ratio = lateral_excursion / max(forward_excursion, 1.0e-12)
        mean_lateral = statistics.mean(
            float(value["plane_normal_dot_lateral_abs"]) for value in valid
        )
        mean_forward = statistics.mean(
            float(value["plane_normal_dot_forward_abs"]) for value in valid
        )
        classification = (
            "sagittal"
            if ratio <= 0.5 and mean_lateral >= 0.8
            else "sideways"
            if ratio >= 0.65 or mean_forward > mean_lateral
            else "ambiguous"
        )
        classifications.append(classification)
        summaries[limb] = {
            "endpoint_forward_excursion_m": forward_excursion,
            "endpoint_lateral_excursion_m": lateral_excursion,
            "lateral_to_forward_excursion_ratio": ratio,
            "valid_plane_frame_count": len(valid),
            "mean_plane_normal_dot_lateral_abs": mean_lateral,
            "mean_plane_normal_dot_forward_abs": mean_forward,
            "classification": classification,
        }

    overall = (
        "four_limb_sagittal_motion"
        if all(value == "sagittal" for value in classifications)
        else "sideways_limb_motion"
        if any(value == "sideways" for value in classifications)
        else "ambiguous_limb_motion"
    )
    return {
        "schema": "shared_four_limb_motion_plane_v1",
        "fps": fps,
        "frame_count": len(frames),
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "limbs": summaries,
        "frames": frame_evidence,
        "overall_classification": overall,
    }


def compute_axial_pose_metrics(
    frames: Sequence[Mapping[str, Sequence[float]]], *, fps: int
) -> dict[str, Any]:
    """Gate pelvis, spine, shoulder girdle, neck, and head lateral roll."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise MotionBasisReviewError("axial-pose FPS must be a positive integer")
    if (
        not isinstance(frames, Sequence)
        or isinstance(frames, (str, bytes))
        or len(frames) < 2
    ):
        raise MotionBasisReviewError("axial-pose review needs at least two frames")
    samples = {name: [] for name in AXIAL_MAXIMUM_ABS_DEGREES}
    evidence = []

    def signed_tilt(vector: np.ndarray, lateral: np.ndarray, description: str) -> float:
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise MotionBasisReviewError(f"{description} vector is invalid")
        return math.degrees(
            math.atan2(float(np.dot(vector, lateral)), float(vector[2]))
        )

    def signed_roll(vector: np.ndarray, lateral: np.ndarray, description: str) -> float:
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise MotionBasisReviewError(f"{description} vector is invalid")
        return math.degrees(
            math.atan2(float(vector[2]), float(np.dot(vector, lateral)))
        )

    for frame_index, frame in enumerate(frames, start=1):
        if not isinstance(frame, Mapping):
            raise MotionBasisReviewError(
                f"axial-pose frame {frame_index} must be a semantic mapping"
            )
        left_shoulder = _point(frame, "left_upper_arm", frame_index)
        right_shoulder = _point(frame, "right_upper_arm", frame_index)
        left_hip = _point(frame, "left_thigh", frame_index)
        right_hip = _point(frame, "right_thigh", frame_index)
        neck = _point(frame, "neck", frame_index)
        head = _point(frame, "head", frame_index)
        head_tail = _point(frame, "head_tail", frame_index)
        lateral = _horizontal_normalized(
            right_shoulder - left_shoulder,
            f"axial-pose frame {frame_index} shoulder lateral basis",
        )
        shoulder_center = 0.5 * (left_shoulder + right_shoulder)
        hip_center = 0.5 * (left_hip + right_hip)
        values = {
            "torso_lateral_tilt_deg": signed_tilt(
                shoulder_center - hip_center, lateral, "torso"
            ),
            "shoulder_roll_deg": signed_roll(
                right_shoulder - left_shoulder, lateral, "shoulder"
            ),
            "hip_roll_deg": signed_roll(
                right_hip - left_hip, lateral, "hip"
            ),
            "head_body_lateral_tilt_deg": signed_tilt(
                head - hip_center, lateral, "head body"
            ),
            "neck_head_lateral_tilt_deg": signed_tilt(
                head - neck, lateral, "neck head"
            ),
            "head_bone_lateral_tilt_deg": signed_tilt(
                head_tail - head, lateral, "head bone"
            ),
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise MotionBasisReviewError(
                    f"axial-pose frame {frame_index} {name} is non-finite"
                )
            samples[name].append(value)
        evidence.append({"frame": frame_index, **values})

    summaries = {}
    passed = True
    for name, values in samples.items():
        maximum_abs = max(abs(value) for value in values)
        limit = AXIAL_MAXIMUM_ABS_DEGREES[name]
        status = "passed" if maximum_abs <= limit else "failed"
        passed = passed and status == "passed"
        summaries[name] = {
            "minimum_deg": min(values),
            "maximum_deg": max(values),
            "mean_abs_deg": statistics.mean(abs(value) for value in values),
            "maximum_abs_deg": maximum_abs,
            "frame_at_maximum_abs": 1
            + max(range(len(values)), key=lambda index: abs(values[index])),
            "maximum_allowed_abs_deg": limit,
            "status": status,
        }
    return {
        "schema": "anatomical_axial_pose_gate_v1",
        "fps": fps,
        "frame_count": len(frames),
        "metrics": summaries,
        "frames": evidence,
        "overall_classification": (
            "axial_pose_within_source_motion_envelope"
            if passed
            else "axial_pose_exceeds_roll_envelope"
        ),
        "automatic_checks": "passed" if passed else "failed",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    path = Path(path)
    root = Path(relative_to)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MotionBasisReviewError(f"artifact is missing, empty, or symlinked: {path}")
    try:
        filename = path.relative_to(root).as_posix()
    except ValueError as error:
        raise MotionBasisReviewError("artifact is outside the review bundle") from error
    return {
        "filename": filename,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionBasisReviewError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise MotionBasisReviewError(f"{description} must be a JSON object")
    return value


def _artifact_path(root: Path, record: Mapping[str, Any], description: str) -> Path:
    filename = record.get("filename")
    if not isinstance(filename, str) or not filename or filename.startswith("/"):
        raise MotionBasisReviewError(f"{description} filename is invalid")
    parts = Path(filename).parts
    if any(part in {"", ".", ".."} or not _SAFE_NAME.fullmatch(part) for part in parts):
        raise MotionBasisReviewError(f"{description} filename is unsafe")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise MotionBasisReviewError(f"{description} is missing or symlinked")
    if record.get("size_bytes") != path.stat().st_size:
        raise MotionBasisReviewError(f"{description} size changed")
    if record.get("sha256") != sha256_file(path):
        raise MotionBasisReviewError(f"{description} SHA-256 changed")
    return path


def validate_review_bundle(bundle_dir: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(bundle_dir)))
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise MotionBasisReviewError("review bundle must be a direct real directory")
    manifest_path = root / "motion_basis_review_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MotionBasisReviewError("review manifest is missing or symlinked")
    manifest = _load_object(manifest_path, "motion-basis review manifest")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("classification") != "technical_diagnostic_only"
        or manifest.get("formal_dataset_asset") is not False
        or manifest.get("decision") != "pending_human_basis_selection"
        or manifest.get("canonical_front") != "negative-y"
        or manifest.get("canonical_up") != "positive-z"
    ):
        raise MotionBasisReviewError("review manifest classification/axis contract is invalid")
    if "user_approved" in json.dumps(manifest, sort_keys=True):
        raise MotionBasisReviewError("motion-basis review may not claim formal approval")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(CANDIDATE_ANGLES):
        raise MotionBasisReviewError("review manifest must contain the exact four candidates")
    expected_files = {"motion_basis_review_manifest.json"}
    invariant_hashes = set()
    for candidate_id, expected_angle in CANDIDATE_ANGLES.items():
        candidate = candidates[candidate_id]
        if not isinstance(candidate, Mapping):
            raise MotionBasisReviewError(f"candidate {candidate_id} is invalid")
        if candidate.get("yaw_degrees") != expected_angle:
            raise MotionBasisReviewError(f"candidate {candidate_id} yaw is invalid")
        matrix = np.asarray(candidate.get("matrix_3x3"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.allclose(
            matrix, yaw_matrix(expected_angle), atol=1.0e-12
        ):
            raise MotionBasisReviewError(f"candidate {candidate_id} matrix is invalid")
        invariant = candidate.get("locked_root_body_sha256")
        if not isinstance(invariant, str) or not re.fullmatch(r"[0-9a-f]{64}", invariant):
            raise MotionBasisReviewError(f"candidate {candidate_id} invariant hash is invalid")
        invariant_hashes.add(invariant)
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping) or metrics.get("schema") != (
            "shared_four_limb_motion_plane_v1"
        ):
            raise MotionBasisReviewError(f"candidate {candidate_id} metrics are invalid")
        artifacts = candidate.get("artifacts")
        expected_artifacts = {"walking.glb", "metrics.json"}
        expected_artifacts.update(f"walking_{view}.mp4" for view in VIEWS)
        expected_artifacts.update(f"walking_{view}.png" for view in VIEWS)
        if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
            raise MotionBasisReviewError(f"candidate {candidate_id} artifact inventory is invalid")
        for label, record in artifacts.items():
            if not isinstance(record, Mapping):
                raise MotionBasisReviewError(f"candidate {candidate_id}/{label} record is invalid")
            path = _artifact_path(root, record, f"candidate {candidate_id}/{label}")
            expected_files.add(path.relative_to(root).as_posix())
    if len(invariant_hashes) != 1:
        raise MotionBasisReviewError("root/body invariants differ across basis candidates")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    actual_directories = [path for path in root.rglob("*") if path.is_dir()]
    if any(path.is_symlink() for path in actual_directories) or actual_files != expected_files:
        raise MotionBasisReviewError("review bundle filesystem inventory is invalid")
    return manifest


def record_selection(
    *,
    bundle_dir: Path,
    selection_dir: Path,
    candidate_id: str | None,
    submitted_manifest_sha256: str,
    reviewer: str,
) -> Path:
    manifest = validate_review_bundle(bundle_dir)
    manifest_path = Path(bundle_dir) / "motion_basis_review_manifest.json"
    current_sha256 = sha256_file(manifest_path)
    if submitted_manifest_sha256 != current_sha256:
        raise MotionBasisReviewError("submitted review snapshot is stale")
    reviewer = str(reviewer).strip()
    if not reviewer:
        raise MotionBasisReviewError("reviewer must be non-empty")
    if candidate_id is not None and candidate_id not in CANDIDATE_ANGLES:
        raise MotionBasisReviewError("selected candidate is not one of the reviewed bases")
    destination = Path(selection_dir)
    if destination.is_symlink():
        raise MotionBasisReviewError("selection directory may not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "retarget_motion_basis_correction_v1.json"
    if output.exists() or output.is_symlink():
        raise MotionBasisReviewError("a motion-basis selection already exists")
    angle = None if candidate_id is None else CANDIDATE_ANGLES[candidate_id]
    payload = {
        "schema": SELECTION_SCHEMA,
        "asset_id": manifest.get("asset_id"),
        "decision": (
            "none_of_the_candidates"
            if candidate_id is None
            else "selected_for_next_retarget"
        ),
        "formal_dataset_asset": False,
        "scope": "bilateral_arm_and_leg_chains_only",
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "candidate_id": candidate_id,
        "yaw_degrees": angle,
        "matrix_3x3": None if angle is None else yaw_matrix(angle).tolist(),
        "candidate_bundle_manifest_sha256": current_sha256,
        "reviewer": reviewer,
        "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination / f".{output.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        output.chmod(0o444)
        directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise MotionBasisReviewError("a motion-basis selection already exists") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
