"""Hash-locked contract and approval gate for FLUX human reference images."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
CANDIDATE_MANIFEST_SCHEMA = "human_reference_candidate_v1"
REFERENCE_REVIEW_SCHEMA = "human_reference_review_v1"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class HumanReferenceNotApproved(RuntimeError):
    """Raised when a human reference review is missing, stale, or rejected."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _reject_symlinked_candidate_dir(candidate_dir: Path) -> Path:
    absolute = Path(candidate_dir).absolute()
    resolved = absolute.resolve()
    if absolute != resolved:
        raise ValueError("candidate directory path must not contain a symlink")
    if not resolved.is_dir():
        raise ValueError(f"candidate directory is missing: {candidate_dir}")
    return resolved


def _regular_file_directly_under(
    path: Path, asset_root: Path, description: str
) -> Path:
    absolute = Path(path).absolute()
    root = Path(asset_root).resolve()
    if not os.path.lexists(absolute):
        raise ValueError(f"{description} is missing: {absolute}")
    resolved = absolute.resolve()
    if absolute != resolved or resolved.parent != root:
        raise ValueError(
            f"{description} must be a regular file directly under the resolved asset root"
        )
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(
            f"{description} must be a regular file directly under the resolved asset root"
        )
    return resolved


def _candidate_image_paths(candidate_dir: Path) -> dict[str, Path]:
    return {
        "source": _regular_file_directly_under(
            candidate_dir / "source.png", candidate_dir, "source image"
        ),
        "candidate": _regular_file_directly_under(
            candidate_dir / "candidate.png", candidate_dir, "candidate image"
        ),
    }


def _read_candidate_snapshot_once(
    candidate_dir: Path,
) -> tuple[bytes, dict[str, bytes], dict[str, Path]]:
    candidate_dir = _reject_symlinked_candidate_dir(Path(candidate_dir))
    manifest_path = _regular_file_directly_under(
        candidate_dir / "candidate_manifest.json", candidate_dir, "candidate manifest"
    )
    images = _candidate_image_paths(candidate_dir)
    try:
        manifest_bytes = manifest_path.read_bytes()
        image_bytes = {name: path.read_bytes() for name, path in images.items()}
    except OSError as error:
        raise ValueError("could not read a stable candidate snapshot") from error
    return manifest_bytes, image_bytes, images


def _validate_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a 64-character lowercase hex value")
    return value


def _validate_expected_snapshot(expected_snapshot: Any) -> dict[str, str]:
    required_fields = {
        "candidate_manifest_sha256",
        "source_sha256",
        "candidate_sha256",
    }
    if not isinstance(expected_snapshot, dict) or set(expected_snapshot) != required_fields:
        raise ValueError(
            "expected_snapshot must contain exactly manifest, source, and candidate hashes"
        )
    return {
        field: _validate_sha256(expected_snapshot[field], f"expected_snapshot {field}")
        for field in required_fields
    }


def _validate_generation_fields(
    *,
    asset_id: Any,
    model_revision: Any,
    prompt: Any,
    seed: Any,
    width: Any,
    height: Any,
    steps: Any,
    guidance_scale: Any,
    source_approval_sha256: Any,
) -> None:
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected human reference asset_id: {asset_id!r}")
    if model_revision != MODEL_REVISION:
        raise ValueError(f"model_revision must be {MODEL_REVISION}")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    for name, value in (("width", width), ("height", height), ("steps", steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(guidance_scale, (int, float))
        or isinstance(guidance_scale, bool)
        or guidance_scale < 0
    ):
        raise ValueError("guidance_scale must be a non-negative number")
    _validate_sha256(source_approval_sha256, "source_approval_sha256")


def write_candidate_manifest(
    candidate_dir: Path,
    *,
    asset_id: str,
    model_revision: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    source_approval_sha256: str,
) -> Path:
    """Atomically record provenance for the source and generated candidate PNGs."""
    candidate_dir = _reject_symlinked_candidate_dir(Path(candidate_dir))
    _validate_generation_fields(
        asset_id=asset_id,
        model_revision=model_revision,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        source_approval_sha256=source_approval_sha256,
    )
    images = _candidate_image_paths(candidate_dir)
    candidate_path = images["candidate"]
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "asset_id": asset_id,
        "model_revision": model_revision,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "source_approval_sha256": source_approval_sha256,
        "input_sha256": sha256_file(images["source"]),
        "output_sha256": sha256_file(candidate_path),
        "output_size_bytes": candidate_path.stat().st_size,
    }
    manifest_path = candidate_dir / "candidate_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def validate_candidate_manifest(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate immutable provenance and current source/candidate image bytes."""
    manifest, images, _ = validated_candidate_snapshot(candidate_dir)
    return manifest, images


def validated_candidate_snapshot(
    candidate_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """Return one validated manifest and image snapshot from stable file bytes."""
    previous: tuple[bytes, dict[str, bytes], dict[str, Path]] | None = None
    stable: tuple[bytes, dict[str, bytes], dict[str, Path]] | None = None
    for _ in range(4):
        current = _read_candidate_snapshot_once(candidate_dir)
        if (
            previous is not None
            and current[0] == previous[0]
            and current[1] == previous[1]
        ):
            stable = current
            break
        previous = current
    if stable is None:
        raise ValueError("candidate snapshot changed while it was being read")

    manifest_bytes, image_bytes, images = stable
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("could not read candidate manifest JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("candidate manifest must contain a JSON object")
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError(
            f"candidate manifest schema_version must be {CANDIDATE_MANIFEST_SCHEMA}"
        )
    _validate_generation_fields(
        asset_id=manifest.get("asset_id"),
        model_revision=manifest.get("model_revision"),
        prompt=manifest.get("prompt"),
        seed=manifest.get("seed"),
        width=manifest.get("width"),
        height=manifest.get("height"),
        steps=manifest.get("steps"),
        guidance_scale=manifest.get("guidance_scale"),
        source_approval_sha256=manifest.get("source_approval_sha256"),
    )
    input_sha256 = _validate_sha256(manifest.get("input_sha256"), "input_sha256")
    output_sha256 = _validate_sha256(manifest.get("output_sha256"), "output_sha256")
    output_size = manifest.get("output_size_bytes")
    if not isinstance(output_size, int) or isinstance(output_size, bool) or output_size < 0:
        raise ValueError("output_size_bytes must be a non-negative integer")

    source_sha256 = hashlib.sha256(image_bytes["source"]).hexdigest()
    candidate_sha256 = hashlib.sha256(image_bytes["candidate"]).hexdigest()
    if source_sha256 != input_sha256:
        raise ValueError("source input image hash does not match candidate manifest")
    if candidate_sha256 != output_sha256:
        raise ValueError("candidate output image hash does not match candidate manifest")
    if len(image_bytes["candidate"]) != output_size:
        raise ValueError("candidate output image size does not match candidate manifest")
    snapshot = {
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256,
    }
    return manifest, images, snapshot


def _existing_review_path(candidate_dir: Path) -> Path | None:
    review_path = Path(candidate_dir) / "reference_review.json"
    if not os.path.lexists(review_path):
        return None
    return _regular_file_directly_under(
        review_path, candidate_dir, "reference review record"
    )


def _pending_payload(
    manifest: dict[str, Any], manifest_sha256: str, image_sha256: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_REVIEW_SCHEMA,
        "asset_id": manifest["asset_id"],
        "decision": "pending",
        "reviewer": "",
        "reviewed_at": None,
        "notes": "",
        "candidate_manifest_sha256": manifest_sha256,
        "source_sha256": image_sha256["source"],
        "candidate_sha256": image_sha256["candidate"],
    }


def _resolve_review_state(candidate_dir: Path) -> tuple[dict[str, Any], bool]:
    manifest, _, snapshot = validated_candidate_snapshot(candidate_dir)
    review_path = _existing_review_path(candidate_dir)
    if review_path is not None:
        existing = _load_json(review_path, "reference review")
        if (
            existing.get("asset_id") == manifest["asset_id"]
            and existing.get("candidate_manifest_sha256")
            == snapshot["candidate_manifest_sha256"]
            and existing.get("source_sha256") == snapshot["source_sha256"]
            and existing.get("candidate_sha256") == snapshot["candidate_sha256"]
        ):
            return existing, True
    return _pending_payload(
        manifest,
        snapshot["candidate_manifest_sha256"],
        {
            "source": snapshot["source_sha256"],
            "candidate": snapshot["candidate_sha256"],
        },
    ), False


def read_review_state(candidate_dir: Path) -> dict[str, Any]:
    """Return the current review or a derived pending state without writing files."""
    state, _ = _resolve_review_state(candidate_dir)
    return state


def record_review(
    candidate_dir: Path,
    decision: str,
    reviewer: str,
    notes: str,
    *,
    expected_snapshot: dict[str, str],
) -> dict[str, Any]:
    """Record a decision only for the exact candidate snapshot the reviewer saw."""
    expected_snapshot = _validate_expected_snapshot(expected_snapshot)
    manifest, _, current_snapshot = validated_candidate_snapshot(candidate_dir)
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    _existing_review_path(candidate_dir)
    if any(
        not hmac.compare_digest(expected_snapshot[field], current_snapshot[field])
        for field in expected_snapshot
    ):
        raise ValueError("candidate snapshot changed; reload before reviewing")
    payload = {
        "schema_version": REFERENCE_REVIEW_SCHEMA,
        "asset_id": manifest["asset_id"],
        "decision": decision,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes.strip(),
        "candidate_manifest_sha256": expected_snapshot[
            "candidate_manifest_sha256"
        ],
        "source_sha256": expected_snapshot["source_sha256"],
        "candidate_sha256": expected_snapshot["candidate_sha256"],
    }
    _atomic_write_json(Path(candidate_dir) / "reference_review.json", payload)
    return payload


def _validate_approved_record(review: dict[str, Any], asset_id: str) -> None:
    if review.get("schema_version") != REFERENCE_REVIEW_SCHEMA:
        raise HumanReferenceNotApproved("reference review schema is invalid")
    if review.get("asset_id") != asset_id:
        raise HumanReferenceNotApproved("reference review asset_id does not match manifest")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise HumanReferenceNotApproved("reference review reviewer must be non-empty")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise HumanReferenceNotApproved(
            "reference review reviewed_at must be timezone-aware ISO-8601"
        )
    try:
        parsed = datetime.fromisoformat(reviewed_at)
    except ValueError as error:
        raise HumanReferenceNotApproved(
            "reference review reviewed_at must be timezone-aware ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HumanReferenceNotApproved(
            "reference review reviewed_at must be timezone-aware ISO-8601"
        )


def assert_reference_approved(candidate_dir: Path) -> dict[str, Any]:
    """Require an approved review record whose hashes still match current files."""
    manifest, _, snapshot = validated_candidate_snapshot(candidate_dir)
    try:
        review_path = _existing_review_path(candidate_dir)
    except ValueError as error:
        raise HumanReferenceNotApproved(str(error)) from error
    if review_path is None:
        raise HumanReferenceNotApproved(f"{manifest['asset_id']} has no reference review")
    try:
        review = _load_json(review_path, "reference review")
    except ValueError as error:
        raise HumanReferenceNotApproved("reference review JSON is invalid") from error
    if review.get("decision") != "approved":
        raise HumanReferenceNotApproved(
            f"{manifest['asset_id']} reference review decision is {review.get('decision')!r}, not approved"
        )
    _validate_approved_record(review, manifest["asset_id"])
    if review.get("source_sha256") != snapshot["source_sha256"]:
        raise HumanReferenceNotApproved(
            f"{manifest['asset_id']} source image hash is stale"
        )
    if review.get("candidate_sha256") != snapshot["candidate_sha256"]:
        raise HumanReferenceNotApproved(
            f"{manifest['asset_id']} candidate image hash is stale"
        )
    if (
        review.get("candidate_manifest_sha256")
        != snapshot["candidate_manifest_sha256"]
    ):
        raise HumanReferenceNotApproved(
            f"{manifest['asset_id']} candidate manifest hash is stale"
        )
    return review


def assert_pair_approved(review_root: Path) -> dict[str, dict[str, Any]]:
    """Require current approvals for the exact male/female reference pair."""
    review_root = Path(review_root)
    root_absolute = review_root.absolute()
    root_resolved = root_absolute.resolve()
    if root_absolute != root_resolved:
        raise HumanReferenceNotApproved("review root path must not contain a symlink")
    if not root_resolved.is_dir():
        raise HumanReferenceNotApproved(f"review root is missing: {review_root}")

    candidate_dirs: dict[str, Path] = {}
    for asset_id in EXPECTED_ASSET_IDS:
        candidate_dir = root_absolute / asset_id
        resolved_candidate_dir = candidate_dir.resolve()
        if candidate_dir != resolved_candidate_dir:
            raise HumanReferenceNotApproved(
                f"{asset_id} candidate directory must not be a symlink"
            )
        try:
            resolved_candidate_dir.relative_to(root_resolved)
        except ValueError as error:
            raise HumanReferenceNotApproved(
                f"{asset_id} candidate directory is outside review root containment"
            ) from error
        if resolved_candidate_dir.name != asset_id:
            raise HumanReferenceNotApproved(
                f"{asset_id} candidate directory resolves to the wrong asset directory"
            )
        if not resolved_candidate_dir.is_dir():
            raise HumanReferenceNotApproved(f"{asset_id} candidate directory is missing")
        candidate_dirs[asset_id] = resolved_candidate_dir

    if len(set(candidate_dirs.values())) != len(EXPECTED_ASSET_IDS):
        raise HumanReferenceNotApproved(
            "expected candidate review directories must be distinct"
        )

    approvals: dict[str, dict[str, Any]] = {}
    for asset_id, candidate_dir in candidate_dirs.items():
        approval = assert_reference_approved(candidate_dir)
        if approval.get("asset_id") != asset_id:
            raise HumanReferenceNotApproved(
                f"{asset_id} approval record asset_id does not match expected asset"
            )
        approvals[asset_id] = approval
    return approvals
