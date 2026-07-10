"""Contract and approval gate for Rocketbox neutral-walk review media."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
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
RETARGET_MANIFEST_SCHEMA = "rocketbox_retarget_manifest_v1"
MOTION_REVIEW_SCHEMA = "rocketbox_motion_review_v1"
REQUIRED_IMMUTABLE_INPUT_HASHES = (
    "avatar_fbx",
    "motion_fbx",
    "source_review",
    "body_color_texture",
    "head_color_texture",
    "opacity_color_texture",
    "retarget_glb",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MotionReviewNotApproved(RuntimeError):
    """Raised when a motion review is missing, stale, rejected, or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temporary: Path | None = None
    fd: int | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            stream = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            fd = None
            raise
        fd = None
        with stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


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


def _reject_symlinked_review_dir(review_dir: Path) -> Path:
    absolute = Path(review_dir).absolute()
    resolved = absolute.resolve()
    if absolute != resolved:
        raise ValueError("review directory path must not contain a symlink")
    return absolute


def _regular_file_directly_under(
    path: Path, asset_root: Path, description: str
) -> Path:
    path = Path(path).absolute()
    root = Path(asset_root).resolve()
    if not os.path.lexists(path):
        raise ValueError(f"{description} is missing: {path}")
    resolved = path.resolve()
    if path != resolved or resolved.parent != root:
        raise ValueError(
            f"{description} must be a regular file directly under the resolved asset root"
        )
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(
            f"{description} must be a regular file directly under the resolved asset root"
        )
    return resolved


def _existing_review_path(review_dir: Path) -> Path | None:
    review_path = Path(review_dir) / "motion_review.json"
    if not os.path.lexists(review_path):
        return None
    return _regular_file_directly_under(
        review_path, review_dir, "motion review record"
    )


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
    review_dir = _reject_symlinked_review_dir(Path(review_dir))
    manifest_path = _regular_file_directly_under(
        review_dir / "retarget_manifest.json", review_dir, "retarget manifest"
    )
    manifest = _load_json(manifest_path, "retarget manifest")

    if manifest.get("schema_version") != RETARGET_MANIFEST_SCHEMA:
        raise ValueError(
            f"retarget manifest schema_version must be {RETARGET_MANIFEST_SCHEMA}"
        )
    asset_id = manifest.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected Rocketbox asset_id: {asset_id!r}")
    input_hashes = manifest.get("immutable_input_hashes")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(
        REQUIRED_IMMUTABLE_INPUT_HASHES
    ):
        raise ValueError(
            "retarget manifest immutable_input_hashes must contain exactly the required keys"
        )
    if any(
        not isinstance(input_hashes[name], str)
        or _SHA256_RE.fullmatch(input_hashes[name]) is None
        for name in REQUIRED_IMMUTABLE_INPUT_HASHES
    ):
        raise ValueError(
            "retarget manifest immutable_input_hashes must contain 64-character lowercase hex values"
        )

    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("retarget manifest binding provenance is required")
    if binding.get("target_asset_id") != asset_id:
        raise ValueError("retarget manifest binding target_asset_id must match asset_id")
    if binding.get("target_mesh_bound") is not True:
        raise ValueError("retarget manifest binding target_mesh_bound must be true")
    if binding.get("official_textures_attached") is not True:
        raise ValueError(
            "retarget manifest binding official_textures_attached must be true"
        )

    media = manifest.get("media")
    if not isinstance(media, dict) or set(media) != set(REQUIRED_MEDIA):
        raise ValueError("retarget manifest media must contain exactly the required media")
    _validate_automatic_checks(manifest.get("automatic_checks"))
    media_paths = {
        name: _media_path(review_dir, name, media[name]) for name in REQUIRED_MEDIA
    }
    return manifest, media_paths


def _current_hashes(review_dir: Path, media_paths: dict[str, Path]) -> tuple[str, dict[str, str]]:
    manifest_path = _regular_file_directly_under(
        Path(review_dir) / "retarget_manifest.json",
        review_dir,
        "retarget manifest",
    )
    return (
        sha256_file(manifest_path),
        {name: sha256_file(path) for name, path in media_paths.items()},
    )


def _pending_payload(
    manifest: dict[str, Any], manifest_sha256: str, media_sha256: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": MOTION_REVIEW_SCHEMA,
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
    review_path = _existing_review_path(review_dir)
    manifest_sha256, media_sha256 = _current_hashes(review_dir, media_paths)
    if review_path is not None:
        existing = _load_json(review_path, "motion review")
        if (
            existing.get("asset_id") == manifest["asset_id"]
            and existing.get("retarget_manifest_sha256") == manifest_sha256
            and existing.get("media_sha256") == media_sha256
        ):
            return existing
    payload = _pending_payload(manifest, manifest_sha256, media_sha256)
    _atomic_write_json(Path(review_dir) / "motion_review.json", payload)
    return payload


def record_decision(
    review_dir: Path, decision: str, reviewer: str, notes: str
) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    _existing_review_path(review_dir)
    manifest_sha256, media_sha256 = _current_hashes(review_dir, media_paths)
    payload = {
        "schema_version": MOTION_REVIEW_SCHEMA,
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


def _validate_approved_record(review: dict[str, Any], asset_id: str) -> None:
    if review.get("schema_version") != MOTION_REVIEW_SCHEMA:
        raise MotionReviewNotApproved("motion review schema is invalid")
    if review.get("asset_id") != asset_id:
        raise MotionReviewNotApproved("motion review asset_id does not match manifest")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise MotionReviewNotApproved("motion review reviewer must be non-empty")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise MotionReviewNotApproved("motion review reviewed_at must be timezone-aware ISO-8601")
    try:
        parsed = datetime.fromisoformat(reviewed_at)
    except ValueError as error:
        raise MotionReviewNotApproved(
            "motion review reviewed_at must be timezone-aware ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MotionReviewNotApproved(
            "motion review reviewed_at must be timezone-aware ISO-8601"
        )


def assert_motion_approved(review_dir: Path) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    try:
        review_path = _existing_review_path(review_dir)
    except ValueError as error:
        raise MotionReviewNotApproved(str(error)) from error
    if review_path is None:
        raise MotionReviewNotApproved(f"{manifest['asset_id']} has no motion review")
    try:
        review = _load_json(review_path, "motion review")
    except ValueError as error:
        raise MotionReviewNotApproved("motion review JSON is invalid") from error
    if review.get("decision") != "approved":
        raise MotionReviewNotApproved(
            f"{manifest['asset_id']} motion review decision is {review.get('decision')!r}, not approved"
        )
    _validate_approved_record(review, manifest["asset_id"])
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
    root_absolute = review_root.absolute()
    root_resolved = root_absolute.resolve()
    if not root_resolved.is_dir():
        raise MotionReviewNotApproved(f"review root is missing: {review_root}")
    approvals: dict[str, dict] = {}
    for asset_id in EXPECTED_ASSET_IDS:
        review_dir = root_absolute / asset_id
        resolved_review_dir = review_dir.resolve()
        try:
            resolved_review_dir.relative_to(root_resolved)
        except ValueError as error:
            raise MotionReviewNotApproved(
                f"{asset_id} review directory is outside review root containment"
            ) from error
        if not review_dir.is_dir():
            raise MotionReviewNotApproved(f"{asset_id} review directory is missing")
        approvals[asset_id] = assert_motion_approved(resolved_review_dir)
    return approvals
