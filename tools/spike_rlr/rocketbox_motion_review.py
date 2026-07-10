"""Contract and approval gate for Rocketbox neutral-walk review media."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MEDIA = (
    "front",
    "side",
    "top",
    "joints",
    "feet",
    "source_target",
    "contact_sheet",
)
EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)


class MotionReviewNotApproved(RuntimeError):
    """Raised when a motion review is missing, stale, rejected, or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _validate_automatic_checks(checks: Any) -> None:
    if not isinstance(checks, dict) or not checks:
        raise ValueError("automatic_checks must be a non-empty object")
    overall = checks.get("overall")
    if overall not in ("passed", True):
        raise ValueError("automatic checks did not pass")
    for value in checks.values():
        if isinstance(value, str) and value in {"failed", "fail", "error"}:
            raise ValueError("automatic checks did not pass")
        if isinstance(value, dict) and value.get("status") in {
            "failed",
            "fail",
            "error",
        }:
            raise ValueError("automatic checks did not pass")


def _media_path(review_dir: Path, name: str, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{name} media path must be a non-empty string")
    root = review_dir.resolve()
    path = (review_dir / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} media path escapes the review directory") from error
    if not path.is_file():
        raise ValueError(f"{name} media file is missing: {path}")
    return path


def validate_ready_manifest(review_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    review_dir = Path(review_dir)
    manifest_path = review_dir / "retarget_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"retarget manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path, "retarget manifest")

    if not isinstance(manifest.get("schema_version"), str) or not manifest[
        "schema_version"
    ].strip():
        raise ValueError("retarget manifest schema_version must be non-empty")
    asset_id = manifest.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected Rocketbox asset_id: {asset_id!r}")
    input_hashes = manifest.get("immutable_input_hashes")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("retarget manifest immutable_input_hashes must be non-empty")
    if any(not isinstance(value, str) or not value for value in input_hashes.values()):
        raise ValueError("retarget manifest immutable_input_hashes must contain values")

    media = manifest.get("media")
    if not isinstance(media, dict) or set(media) != set(REQUIRED_MEDIA):
        raise ValueError("retarget manifest media must contain exactly the required media")
    _validate_automatic_checks(manifest.get("automatic_checks"))
    media_paths = {
        name: _media_path(review_dir, name, media[name]) for name in REQUIRED_MEDIA
    }
    return manifest, media_paths


def _current_hashes(review_dir: Path, media_paths: dict[str, Path]) -> tuple[str, dict[str, str]]:
    return (
        sha256_file(Path(review_dir) / "retarget_manifest.json"),
        {name: sha256_file(path) for name, path in media_paths.items()},
    )


def _pending_payload(
    manifest: dict[str, Any], manifest_sha256: str, media_sha256: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "rocketbox_motion_review_v1",
        "asset_id": manifest["asset_id"],
        "decision": "pending",
        "reviewer": "",
        "reviewed_at": None,
        "notes": "",
        "retarget_manifest_sha256": manifest_sha256,
        "media_sha256": media_sha256,
    }


def ensure_pending_review(review_dir: Path) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    review_path = Path(review_dir) / "motion_review.json"
    manifest_sha256, media_sha256 = _current_hashes(review_dir, media_paths)
    if review_path.is_file():
        existing = _load_json(review_path, "motion review")
        if (
            existing.get("asset_id") == manifest["asset_id"]
            and existing.get("retarget_manifest_sha256") == manifest_sha256
            and existing.get("media_sha256") == media_sha256
        ):
            return existing
    payload = _pending_payload(manifest, manifest_sha256, media_sha256)
    _atomic_write_json(review_path, payload)
    return payload


def record_decision(
    review_dir: Path, decision: str, reviewer: str, notes: str
) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    manifest_sha256, media_sha256 = _current_hashes(review_dir, media_paths)
    payload = {
        "schema_version": "rocketbox_motion_review_v1",
        "asset_id": manifest["asset_id"],
        "decision": decision,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes.strip(),
        "retarget_manifest_sha256": manifest_sha256,
        "media_sha256": media_sha256,
    }
    _atomic_write_json(Path(review_dir) / "motion_review.json", payload)
    return payload


def assert_motion_approved(review_dir: Path) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    review_path = Path(review_dir) / "motion_review.json"
    if not review_path.is_file():
        raise MotionReviewNotApproved(f"{manifest['asset_id']} has no motion review")
    review = _load_json(review_path, "motion review")
    if review.get("decision") != "approved":
        raise MotionReviewNotApproved(
            f"{manifest['asset_id']} motion review decision is {review.get('decision')!r}, not approved"
        )
    manifest_sha256, media_sha256 = _current_hashes(review_dir, media_paths)
    if review.get("retarget_manifest_sha256") != manifest_sha256:
        raise MotionReviewNotApproved(
            f"{manifest['asset_id']} retarget manifest hash is stale"
        )
    recorded_media = review.get("media_sha256")
    if not isinstance(recorded_media, dict):
        raise MotionReviewNotApproved(f"{manifest['asset_id']} media hashes are missing")
    for name in REQUIRED_MEDIA:
        if recorded_media.get(name) != media_sha256[name]:
            raise MotionReviewNotApproved(f"{name} media hash is stale")
    return review


def assert_pair_approved(review_root: Path) -> dict[str, dict]:
    review_root = Path(review_root)
    approvals: dict[str, dict] = {}
    for asset_id in EXPECTED_ASSET_IDS:
        review_dir = review_root / asset_id
        if not review_dir.is_dir():
            raise MotionReviewNotApproved(f"{asset_id} review directory is missing")
        approvals[asset_id] = assert_motion_approved(review_dir)
    return approvals
