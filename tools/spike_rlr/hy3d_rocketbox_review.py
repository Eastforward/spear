"""Hash-locked contract and paired gate for Hunyuan/Rocketbox Walk/Idle review."""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
REQUIRED_MOTIONS = ("walk", "idle")
REQUIRED_VIEWS = ("front", "side", "feet")
BIND_MANIFEST_SCHEMA = "hy3d_rocketbox_bind_v1"
REVIEW_MANIFEST_SCHEMA = "hy3d_rocketbox_review_manifest_v1"
DIRECT_ATTEMPT_READY_SCHEMA = "hy3d_rocketbox_direct_attempt_ready_v1"
PIXEL_QA_SCHEMA = "hy3d_rocketbox_pixel_qa_v1"
ARTIFACT_SNAPSHOT_SCHEMA = "hy3d_rocketbox_artifact_snapshot_v1"
DECISION_SCHEMA = "hy3d_rocketbox_review_v1"
PIXEL_QA_CHECKS = (
    "hands_attached",
    "hands_not_duplicated",
    "pieces_nonblank",
    "arm_torso_regions_clean",
    "thigh_regions_clean",
    "sleeves_seam_free",
    "feet_not_inverted",
    "floor_cards_absent",
    "leg_gap_fans_absent",
    "mesh_explosions_absent",
)
SNAPSHOT_FIELDS = (
    "bind_manifest_sha256",
    "review_manifest_sha256",
    "direct_attempt_ready_sha256",
    "pixel_qa_sha256",
    "reference_sha256",
    "bound_blend_sha256",
    "bind_metrics_sha256",
    "bind_contact_sheet_sha256",
    "walk_glb_sha256",
    "idle_glb_sha256",
    "walk_front_sha256",
    "walk_side_sha256",
    "walk_feet_sha256",
    "idle_front_sha256",
    "idle_side_sha256",
    "idle_feet_sha256",
)
_ARTIFACT_SNAPSHOT_FIELDS = {
    "schema_version",
    "asset_id",
    "bind_manifest_sha256",
    "review_manifest_sha256",
    "bound_blend",
    "glbs",
    "videos",
    "bind_metrics",
    "contact_sheet",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_DECISION_FILENAME = "hy3d_rocketbox_review.json"


class Hy3DRocketboxNotApproved(RuntimeError):
    """Raised when a Walk/Idle review is absent, stale, or rejected."""


def _open_directory_fd(path: Path, description: str) -> int:
    """Open every directory component without following a symlink."""
    absolute = Path(path).absolute()
    components = absolute.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{description} must not contain path traversal")
    try:
        directory_fd = os.open(os.path.sep, _DIRECTORY_FLAGS)
    except OSError as error:
        raise ValueError(f"could not open filesystem root for {description}") from error
    try:
        for component in components:
            child_fd = os.open(
                component,
                _DIRECTORY_FLAGS,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise ValueError(f"{description} must be a directory")
        return directory_fd
    except (OSError, ValueError) as error:
        os.close(directory_fd)
        if isinstance(error, ValueError):
            raise
        raise ValueError(
            f"{description} must not be a symlink and must be an existing directory"
        ) from error


def _open_child_directory_fd(parent_fd: int, name: str, description: str) -> int:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"{description} is not a direct child")
    child_fd: int | None = None
    try:
        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
            raise OSError(errno.ENOTDIR, "not a directory")
        return child_fd
    except OSError as error:
        if child_fd is not None:
            os.close(child_fd)
        raise ValueError(
            f"{description} must not be a symlink and must be an existing directory"
        ) from error


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file_at(
    directory_fd: int,
    filename: str,
    description: str,
    *,
    missing_ok: bool = False,
) -> bytes | None:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"{description} must be directly under the asset root")
    try:
        file_fd = os.open(filename, _READ_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError as error:
        if missing_ok:
            return None
        raise ValueError(f"{description} is missing: {filename}") from error
    except OSError as error:
        raise ValueError(
            f"{description} must be a non-symlink regular file under the asset root"
        ) from error
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{description} must be a regular file")
        if before.st_size <= 0:
            raise ValueError(f"{description} must be non-empty")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        value = b"".join(chunks)
        if _stat_signature(before) != _stat_signature(after) or len(value) != after.st_size:
            raise ValueError(f"{description} changed while it was being read")
        return value
    except OSError as error:
        raise ValueError(f"could not read stable {description} bytes") from error
    finally:
        os.close(file_fd)


def _read_file_from_parent(path: Path, description: str) -> bytes:
    absolute = Path(path).absolute()
    directory_fd = _open_directory_fd(absolute.parent, f"{description} parent directory")
    try:
        value = _read_regular_file_at(directory_fd, absolute.name, description)
        assert value is not None
        return value
    finally:
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    """Hash a regular non-empty file opened relative to its parent directory FD."""
    return hashlib.sha256(_read_file_from_parent(Path(path), "hash input")).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path = Path(path).absolute()
    directory_fd = _open_directory_fd(path.parent, "decision parent directory")
    temporary_name = ""
    temporary_fd: int | None = None
    try:
        for _ in range(16):
            temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
            try:
                temporary_fd = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory_fd,
                )
                break
            except FileExistsError:
                continue
        if temporary_fd is None:
            raise OSError("could not allocate a decision staging file")
        offset = 0
        while offset < len(value):
            written = os.write(temporary_fd, value[offset:])
            if written <= 0:
                raise OSError("could not write complete decision bytes")
            offset += written
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        os.fsync(directory_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _remove_decision(asset_dir: Path) -> None:
    directory_fd = _open_directory_fd(Path(asset_dir), "asset directory")
    try:
        try:
            os.unlink(_DECISION_FILENAME, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_object(raw: bytes, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _reject_if_direct_attempt_rejected(directory_fd: int) -> None:
    try:
        rejected_fd = os.open(
            "direct_attempt_rejected.json",
            _READ_FLAGS,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("direct attempt is rejected") from error
    else:
        os.close(rejected_fd)
        raise ValueError("direct attempt is rejected")


def _read_review_snapshot_once(asset_dir: Path) -> dict[str, bytes]:
    directory_fd = _open_directory_fd(Path(asset_dir), "asset directory")
    try:
        _reject_if_direct_attempt_rejected(directory_fd)
        names = {
            "bind_manifest": ("bind_manifest.json", "bind manifest"),
            "review_manifest": ("review_manifest.json", "review manifest"),
            "direct_attempt_ready": (
                "direct_attempt_ready.json",
                "direct attempt ready record",
            ),
            "pixel_qa": ("pixel_qa.json", "pixel QA"),
            "reference": ("reference.png", "approved FLUX reference"),
            "bound_blend": ("bound.blend", "bound blend"),
            "bind_metrics": ("bind_metrics.json", "bind metrics"),
            "contact_sheet": ("bind_contact_sheet.png", "contact sheet"),
            "walk_glb": ("bound_walk.glb", "bound walk GLB"),
            "idle_glb": ("bound_idle.glb", "bound idle GLB"),
        }
        for motion in REQUIRED_MOTIONS:
            for view in REQUIRED_VIEWS:
                names[f"{motion}_{view}"] = (
                    f"{motion}_{view}.mp4",
                    f"{motion} {view} video",
                )
        result: dict[str, bytes] = {}
        for key, (filename, description) in names.items():
            value = _read_regular_file_at(directory_fd, filename, description)
            assert value is not None
            result[key] = value
        return result
    finally:
        os.close(directory_fd)


def _validate_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a 64-character lowercase hex value")
    return value


def _validate_file_descriptor(
    value: Any, filename: str, actual: bytes, description: str
) -> None:
    if not isinstance(value, dict) or set(value) != {"filename", "sha256"}:
        raise ValueError(f"{description} must contain exactly filename and sha256")
    if value["filename"] != filename:
        raise ValueError(f"{description} must use canonical media filename {filename}")
    expected = _validate_sha256(value["sha256"], f"{description} sha256")
    if not hmac.compare_digest(expected, hashlib.sha256(actual).hexdigest()):
        raise ValueError(f"{description} hash does not match current file")


def _validate_glbs(value: Any, data: dict[str, bytes], description: str) -> None:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_MOTIONS):
        raise ValueError(f"{description} must contain exactly walk and idle")
    for motion in REQUIRED_MOTIONS:
        _validate_file_descriptor(
            value[motion],
            f"bound_{motion}.glb",
            data[f"{motion}_glb"],
            f"{description} bound {motion} GLB",
        )


def _validate_videos(value: Any, data: dict[str, bytes], description: str) -> None:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_MOTIONS):
        raise ValueError(f"{description} videos must contain exactly walk and idle")
    for motion in REQUIRED_MOTIONS:
        views = value[motion]
        if not isinstance(views, dict) or set(views) != set(REQUIRED_VIEWS):
            raise ValueError(
                f"{description} {motion} videos must contain exactly front, side, and feet"
            )
        for view in REQUIRED_VIEWS:
            _validate_file_descriptor(
                views[view],
                f"{motion}_{view}.mp4",
                data[f"{motion}_{view}"],
                f"{description} {motion} {view} video",
            )


def _validate_action_names(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_MOTIONS):
        raise ValueError("bind manifest action_names must contain exactly walk and idle")
    if any(
        not isinstance(value[motion], str) or not value[motion]
        for motion in REQUIRED_MOTIONS
    ):
        raise ValueError("bind manifest action names must be non-empty strings")
    if value["walk"] == value["idle"]:
        raise ValueError("bind manifest walk and idle action names must be different")


def _validate_expected_artifact_snapshot(
    expected: Any,
    ready: dict[str, Any],
    asset_id: str,
    data: dict[str, bytes],
) -> None:
    if not isinstance(expected, dict):
        raise ValueError("pixel QA expected artifact snapshot is missing")
    if set(expected) != _ARTIFACT_SNAPSHOT_FIELDS:
        raise ValueError("pixel QA expected artifact snapshot must be complete")
    if expected.get("schema_version") != ARTIFACT_SNAPSHOT_SCHEMA:
        raise ValueError("pixel QA expected artifact snapshot schema is invalid")
    if expected.get("asset_id") != asset_id:
        raise ValueError("pixel QA expected artifact snapshot asset_id is stale")
    try:
        bind_hash = _validate_sha256(
            expected.get("bind_manifest_sha256"),
            "pixel QA expected bind manifest sha256",
        )
        review_hash = _validate_sha256(
            expected.get("review_manifest_sha256"),
            "pixel QA expected review manifest sha256",
        )
        if not hmac.compare_digest(bind_hash, hashlib.sha256(data["bind_manifest"]).hexdigest()):
            raise ValueError
        if not hmac.compare_digest(
            review_hash, hashlib.sha256(data["review_manifest"]).hexdigest()
        ):
            raise ValueError
        if bind_hash != ready.get("bind_manifest_sha256"):
            raise ValueError
        if review_hash != ready.get("review_manifest_sha256"):
            raise ValueError
        if expected.get("bound_blend") != ready.get("bound_blend"):
            raise ValueError
        if expected.get("glbs") != ready.get("glbs"):
            raise ValueError
        if expected.get("videos") != ready.get("videos"):
            raise ValueError
        if expected.get("bind_metrics") != ready.get("bind_metrics"):
            raise ValueError
        if expected.get("contact_sheet") != ready.get("contact_sheet"):
            raise ValueError
        _validate_file_descriptor(
            expected.get("bound_blend"),
            "bound.blend",
            data["bound_blend"],
            "pixel QA expected bound blend",
        )
        _validate_glbs(expected.get("glbs"), data, "pixel QA expected GLBs")
        _validate_videos(expected.get("videos"), data, "pixel QA expected snapshot")
        _validate_file_descriptor(
            expected.get("bind_metrics"),
            "bind_metrics.json",
            data["bind_metrics"],
            "pixel QA expected bind metrics",
        )
        _validate_file_descriptor(
            expected.get("contact_sheet"),
            "bind_contact_sheet.png",
            data["contact_sheet"],
            "pixel QA expected contact sheet",
        )
    except (KeyError, ValueError) as error:
        raise ValueError("pixel QA expected artifact snapshot is stale") from error


def _validate_pixel_qa(
    value: dict[str, Any],
    ready: dict[str, Any],
    asset_id: str,
    data: dict[str, bytes],
) -> None:
    if value.get("schema_version") != PIXEL_QA_SCHEMA:
        raise ValueError(f"pixel QA schema_version must be {PIXEL_QA_SCHEMA}")
    if value.get("asset_id") != asset_id:
        raise ValueError("pixel QA asset_id must match bind manifest asset_id")
    if value.get("decision") != "ready":
        raise ValueError("pixel QA decision must be ready")
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise ValueError("pixel QA reviewer must be non-empty")
    if not isinstance(value.get("reviewed_at"), str) or not value["reviewed_at"].strip():
        raise ValueError("pixel QA reviewed_at must be non-empty")
    if not isinstance(value.get("notes", ""), str):
        raise ValueError("pixel QA notes must be text")
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(PIXEL_QA_CHECKS):
        raise ValueError("pixel QA checks are incomplete")
    if any(checks[check] is not True for check in PIXEL_QA_CHECKS):
        raise ValueError("ready pixel QA requires every visual check to pass")
    _validate_expected_artifact_snapshot(
        value.get("expected_artifact_snapshot"), ready, asset_id, data
    )


def _validate_ready_record(
    ready: dict[str, Any],
    asset_id: str,
    data: dict[str, bytes],
) -> None:
    required_fields = {
        "schema_version",
        "asset_id",
        "status",
        "bind_manifest_sha256",
        "review_manifest_sha256",
        "bound_blend",
        "glbs",
        "videos",
        "pixel_qa",
        "bind_metrics",
        "contact_sheet",
    }
    if not required_fields.issubset(ready):
        raise ValueError("direct attempt ready record is missing required fields")
    if ready.get("schema_version") != DIRECT_ATTEMPT_READY_SCHEMA:
        raise ValueError(
            f"direct attempt ready schema_version must be {DIRECT_ATTEMPT_READY_SCHEMA}"
        )
    if ready.get("asset_id") != asset_id:
        raise ValueError("direct attempt ready asset_id must match bind manifest asset_id")
    if ready.get("status") != "ready":
        raise ValueError("direct attempt ready status must be ready")
    for field, data_key, description in (
        ("bind_manifest_sha256", "bind_manifest", "bind manifest"),
        ("review_manifest_sha256", "review_manifest", "review manifest"),
    ):
        expected = _validate_sha256(ready.get(field), f"direct attempt ready {field}")
        actual = hashlib.sha256(data[data_key]).hexdigest()
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"direct attempt ready {description} hash is stale")
    _validate_file_descriptor(
        ready.get("bound_blend"),
        "bound.blend",
        data["bound_blend"],
        "direct attempt ready bound blend",
    )
    try:
        _validate_glbs(ready.get("glbs"), data, "direct attempt ready GLBs")
        _validate_videos(ready.get("videos"), data, "direct attempt ready")
    except ValueError as error:
        raise ValueError(f"direct attempt ready record: {error}") from error
    _validate_file_descriptor(
        ready.get("pixel_qa"),
        "pixel_qa.json",
        data["pixel_qa"],
        "direct attempt ready pixel QA",
    )
    _validate_file_descriptor(
        ready.get("bind_metrics"),
        "bind_metrics.json",
        data["bind_metrics"],
        "direct attempt ready bind metrics",
    )
    _validate_file_descriptor(
        ready.get("contact_sheet"),
        "bind_contact_sheet.png",
        data["contact_sheet"],
        "direct attempt ready contact sheet",
    )
    pixel_qa = _load_object(data["pixel_qa"], "pixel QA")
    _validate_pixel_qa(pixel_qa, ready, asset_id, data)


def _validate_snapshot(
    data: dict[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    bind_manifest = _load_object(data["bind_manifest"], "bind manifest")
    review_manifest = _load_object(data["review_manifest"], "review manifest")
    ready = _load_object(data["direct_attempt_ready"], "direct attempt ready record")
    if bind_manifest.get("schema_version") != BIND_MANIFEST_SCHEMA:
        raise ValueError(f"bind manifest schema_version must be {BIND_MANIFEST_SCHEMA}")
    asset_id = bind_manifest.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected Rocketbox asset_id: {asset_id!r}")
    _validate_file_descriptor(
        bind_manifest.get("reference"),
        "reference.png",
        data["reference"],
        "reference",
    )
    _validate_file_descriptor(
        bind_manifest.get("bound_blend"),
        "bound.blend",
        data["bound_blend"],
        "bound blend",
    )
    _validate_glbs(bind_manifest.get("glbs"), data, "bind manifest GLBs")
    _validate_action_names(bind_manifest.get("action_names"))
    artifacts = bind_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("bind manifest artifacts must be an object")
    _validate_file_descriptor(
        artifacts.get("bind_metrics"),
        "bind_metrics.json",
        data["bind_metrics"],
        "bind manifest bind metrics",
    )

    if review_manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
        raise ValueError(
            f"review manifest schema_version must be {REVIEW_MANIFEST_SCHEMA}"
        )
    if review_manifest.get("asset_id") != asset_id:
        raise ValueError("review manifest asset_id must match bind manifest asset_id")
    bind_hash = hashlib.sha256(data["bind_manifest"]).hexdigest()
    if not hmac.compare_digest(
        _validate_sha256(
            review_manifest.get("bind_manifest_sha256"),
            "review manifest bind_manifest_sha256",
        ),
        bind_hash,
    ):
        raise ValueError(
            "review manifest bind manifest hash does not match current bind manifest"
        )
    _validate_glbs(review_manifest.get("glbs"), data, "review manifest GLBs")
    _validate_videos(review_manifest.get("videos"), data, "review manifest")
    _validate_ready_record(ready, asset_id, data)
    snapshot = {
        "bind_manifest_sha256": bind_hash,
        "review_manifest_sha256": hashlib.sha256(data["review_manifest"]).hexdigest(),
        "direct_attempt_ready_sha256": hashlib.sha256(
            data["direct_attempt_ready"]
        ).hexdigest(),
        "pixel_qa_sha256": hashlib.sha256(data["pixel_qa"]).hexdigest(),
        "reference_sha256": hashlib.sha256(data["reference"]).hexdigest(),
        "bound_blend_sha256": hashlib.sha256(data["bound_blend"]).hexdigest(),
        "bind_metrics_sha256": hashlib.sha256(data["bind_metrics"]).hexdigest(),
        "bind_contact_sheet_sha256": hashlib.sha256(
            data["contact_sheet"]
        ).hexdigest(),
    }
    for motion in REQUIRED_MOTIONS:
        snapshot[f"{motion}_glb_sha256"] = hashlib.sha256(
            data[f"{motion}_glb"]
        ).hexdigest()
        for view in REQUIRED_VIEWS:
            snapshot[f"{motion}_{view}_sha256"] = hashlib.sha256(
                data[f"{motion}_{view}"]
            ).hexdigest()
    return bind_manifest, review_manifest, snapshot


def validated_review_snapshot(
    asset_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, bytes],
    dict[str, str],
]:
    """Return parsed manifests, immutable bytes, and hashes from one stable read."""
    previous: dict[str, bytes] | None = None
    stable: dict[str, bytes] | None = None
    for _ in range(4):
        current = _read_review_snapshot_once(asset_dir)
        if previous is not None and current == previous:
            stable = current
            break
        previous = current
    if stable is None:
        raise ValueError("review snapshot changed while it was being read")
    bind_manifest, review_manifest, snapshot = _validate_snapshot(stable)
    return bind_manifest, review_manifest, MappingProxyType(stable.copy()), snapshot


def _read_decision_bytes(asset_dir: Path) -> bytes | None:
    directory_fd = _open_directory_fd(Path(asset_dir), "asset directory")
    try:
        return _read_regular_file_at(
            directory_fd,
            _DECISION_FILENAME,
            "review decision",
            missing_ok=True,
        )
    finally:
        os.close(directory_fd)


def _pending_payload(asset_id: str, snapshot: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA,
        "asset_id": asset_id,
        "decision": "pending",
        "reviewer": "",
        "reviewed_at": None,
        "notes": "",
        "snapshot": snapshot,
    }


def read_review_state_for_snapshot(
    asset_dir: Path,
    bind_manifest: dict[str, Any],
    snapshot: dict[str, str],
) -> dict[str, Any]:
    """Resolve a decision against an already captured artifact snapshot."""
    asset_id = bind_manifest.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected Rocketbox asset_id: {asset_id!r}")
    snapshot = _validate_expected_snapshot(snapshot)
    decision_bytes = _read_decision_bytes(asset_dir)
    if decision_bytes is not None:
        decision = _load_object(decision_bytes, "review decision")
        if decision.get("asset_id") == asset_id and decision.get("snapshot") == snapshot:
            return decision
    return _pending_payload(asset_id, snapshot)


def read_review_state(asset_dir: Path) -> dict[str, Any]:
    """Return the current decision or a derived pending state without writing."""
    bind_manifest, _, _, snapshot = validated_review_snapshot(asset_dir)
    return read_review_state_for_snapshot(asset_dir, bind_manifest, snapshot)


def _validate_expected_snapshot(expected_snapshot: Any) -> dict[str, str]:
    if not isinstance(expected_snapshot, dict) or set(expected_snapshot) != set(
        SNAPSHOT_FIELDS
    ):
        raise ValueError("expected_snapshot must contain exactly all bound snapshot hashes")
    return {
        field: _validate_sha256(expected_snapshot[field], f"expected_snapshot {field}")
        for field in SNAPSHOT_FIELDS
    }


def assert_snapshot_current(
    asset_dir: Path, expected_snapshot: dict[str, str]
) -> None:
    """Fail if a captured page/media snapshot is no longer fully ready and current."""
    expected_snapshot = _validate_expected_snapshot(expected_snapshot)
    _, _, _, current_snapshot = validated_review_snapshot(asset_dir)
    if any(
        not hmac.compare_digest(expected_snapshot[field], current_snapshot[field])
        for field in SNAPSHOT_FIELDS
    ):
        raise ValueError("review snapshot changed; reload before reviewing")


def record_decision(
    asset_dir: Path,
    decision: str,
    reviewer: str,
    notes: str,
    *,
    expected_snapshot: dict[str, str],
) -> dict[str, Any]:
    """Atomically record a decision only for the snapshot a reviewer inspected."""
    expected_snapshot = _validate_expected_snapshot(expected_snapshot)
    bind_manifest, _, _, snapshot = validated_review_snapshot(asset_dir)
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    if any(
        not hmac.compare_digest(expected_snapshot[field], snapshot[field])
        for field in SNAPSHOT_FIELDS
    ):
        raise ValueError("review snapshot changed; reload before reviewing")
    previous_decision = _read_decision_bytes(asset_dir)
    payload = {
        "schema_version": DECISION_SCHEMA,
        "asset_id": bind_manifest["asset_id"],
        "decision": decision,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes.strip() if isinstance(notes, str) else "",
        "snapshot": snapshot,
    }
    decision_path = Path(asset_dir) / _DECISION_FILENAME
    _atomic_write_json(decision_path, payload)
    try:
        assert_snapshot_current(asset_dir, snapshot)
    except BaseException:
        if previous_decision is None:
            _remove_decision(asset_dir)
        else:
            _atomic_write_bytes(decision_path, previous_decision)
        raise
    return payload


def _validate_approved_decision(decision: dict[str, Any], asset_id: str) -> None:
    if decision.get("schema_version") != DECISION_SCHEMA:
        raise Hy3DRocketboxNotApproved("review decision schema is invalid")
    if decision.get("asset_id") != asset_id:
        raise Hy3DRocketboxNotApproved(
            "review decision asset_id does not match manifest"
        )
    if not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip():
        raise Hy3DRocketboxNotApproved("review decision reviewer must be non-empty")
    reviewed_at = decision.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise Hy3DRocketboxNotApproved("reviewed_at must be timezone-aware ISO-8601")
    try:
        parsed = datetime.fromisoformat(reviewed_at)
    except ValueError as error:
        raise Hy3DRocketboxNotApproved(
            "reviewed_at must be timezone-aware ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Hy3DRocketboxNotApproved("reviewed_at must be timezone-aware ISO-8601")


def _capture_approved_asset(asset_dir: Path) -> dict[str, Any]:
    bind_manifest, _, captured, snapshot = validated_review_snapshot(asset_dir)
    decision_bytes = _read_decision_bytes(asset_dir)
    if decision_bytes is None:
        raise Hy3DRocketboxNotApproved(
            f"{bind_manifest['asset_id']} has no review decision"
        )
    decision = _load_object(decision_bytes, "review decision")
    if decision.get("decision") != "approved":
        raise Hy3DRocketboxNotApproved(
            f"{bind_manifest['asset_id']} review decision is "
            f"{decision.get('decision')!r}, not approved"
        )
    _validate_approved_decision(decision, bind_manifest["asset_id"])
    if decision.get("snapshot") != snapshot:
        raise Hy3DRocketboxNotApproved("review snapshot is stale")
    return {
        "bind_asset_id": bind_manifest["asset_id"],
        "decision": decision,
        "decision_bytes": decision_bytes,
        "snapshot": snapshot,
        "captured": dict(captured),
    }


def assert_asset_approved(asset_dir: Path) -> dict[str, Any]:
    try:
        first = _capture_approved_asset(asset_dir)
        second = _capture_approved_asset(asset_dir)
    except Hy3DRocketboxNotApproved:
        raise
    except ValueError as error:
        raise Hy3DRocketboxNotApproved(str(error)) from error
    if first != second:
        raise Hy3DRocketboxNotApproved(
            "review snapshot changed while approval was being read"
        )
    return first["decision"]


def _validate_pair_directories(review_root: Path) -> dict[str, Path]:
    root = Path(review_root).absolute()
    try:
        root_fd = _open_directory_fd(root, "review root")
    except ValueError as error:
        raise Hy3DRocketboxNotApproved(str(error)) from error
    identities: set[tuple[int, int]] = set()
    directories: dict[str, Path] = {}
    try:
        for asset_id in EXPECTED_ASSET_IDS:
            try:
                child_fd = _open_child_directory_fd(
                    root_fd, asset_id, f"{asset_id} review directory"
                )
            except ValueError as error:
                raise Hy3DRocketboxNotApproved(str(error)) from error
            try:
                child_stat = os.fstat(child_fd)
                identity = (child_stat.st_dev, child_stat.st_ino)
                if identity in identities:
                    raise Hy3DRocketboxNotApproved(
                        "expected review directories must be distinct"
                    )
                identities.add(identity)
                directories[asset_id] = root / asset_id
            finally:
                os.close(child_fd)
    finally:
        os.close(root_fd)
    return directories


def _validate_capture_for_directory_slot(
    expected_asset_id: str, capture: dict[str, Any]
) -> None:
    if capture.get("bind_asset_id") != expected_asset_id:
        raise Hy3DRocketboxNotApproved(
            f"{expected_asset_id} directory slot bind asset_id is "
            f"{capture.get('bind_asset_id')!r}"
        )
    decision = capture.get("decision")
    decision_asset_id = decision.get("asset_id") if isinstance(decision, dict) else None
    if decision_asset_id != expected_asset_id:
        raise Hy3DRocketboxNotApproved(
            f"{expected_asset_id} directory slot decision asset_id is "
            f"{decision_asset_id!r}"
        )


def _capture_approved_pair(
    directories: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for expected_asset_id in EXPECTED_ASSET_IDS:
        asset_capture = _capture_approved_asset(directories[expected_asset_id])
        _validate_capture_for_directory_slot(expected_asset_id, asset_capture)
        captured[expected_asset_id] = asset_capture
    return captured


def assert_pair_approved(review_root: Path) -> dict[str, dict[str, Any]]:
    directories = _validate_pair_directories(review_root)
    try:
        first = _capture_approved_pair(directories)
        second = _capture_approved_pair(directories)
    except Hy3DRocketboxNotApproved:
        raise
    except ValueError as error:
        raise Hy3DRocketboxNotApproved(str(error)) from error
    if first != second:
        raise Hy3DRocketboxNotApproved(
            "pair snapshot changed while approvals were being read"
        )
    return {
        asset_id: first[asset_id]["decision"] for asset_id in EXPECTED_ASSET_IDS
    }
