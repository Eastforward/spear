#!/usr/bin/env python3
"""Freeze objective machine-visible defects for one exact derived-static review.

This tool does not create a user or human decision.  It publishes one
immutable fail-closed receipt that binds an externally pinned review JSON, its
internal self-hash, the preserved raw rejection, enumerated objective defect
codes, and exact visual evidence artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import (
    controlled_animal_derived_static_machine_rejection_contract as receipt_contract,
)
from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts

SPEAR_ROOT = Path(__file__).resolve().parents[1]
LOGICAL_TMP_ROOT = SPEAR_ROOT / "tmp"
PHYSICAL_TMP_ROOT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ARTIFACT_PATHS = {
    "clay_back": ("evidence", "clay_five_view", "views", "back"),
    "clay_contact_sheet": ("evidence", "clay_five_view", "contact_sheet"),
    "clay_front": ("evidence", "clay_five_view", "views", "front"),
    "clay_quarter": ("evidence", "clay_five_view", "views", "quarter"),
    "clay_side": ("evidence", "clay_five_view", "views", "side"),
    "clay_top": ("evidence", "clay_five_view", "views", "top"),
    "pbr_back": ("evidence", "pbr_five_view", "views", "back"),
    "pbr_contact_sheet": ("evidence", "pbr_five_view", "contact_sheet"),
    "pbr_front": ("evidence", "pbr_five_view", "views", "front"),
    "pbr_quarter": ("evidence", "pbr_five_view", "views", "quarter"),
    "pbr_side": ("evidence", "pbr_five_view", "views", "side"),
    "pbr_top": ("evidence", "pbr_five_view", "views", "top"),
}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _path_components(path: Path) -> list[Path]:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    result: list[Path] = []
    for part in absolute.parts[1:]:
        current = current / part
        result.append(current)
    return result


def _allowed_tmp_bridge(component: Path) -> bool:
    if component != LOGICAL_TMP_ROOT or not component.is_symlink():
        return False
    try:
        target = Path(os.readlink(component))
    except OSError:
        return False
    if not target.is_absolute():
        target = component.parent / target
    try:
        return os.path.samefile(target, PHYSICAL_TMP_ROOT)
    except OSError:
        return False


def _physical_input_path(path: Path, label: str) -> Path:
    absolute = Path(path).absolute()
    components = _path_components(absolute)
    for index, component in enumerate(components):
        if not component.is_symlink():
            continue
        if _allowed_tmp_bridge(component) and index != len(components) - 1:
            continue
        raise contracts.ContractError(
            f"{label} contains an unsafe symlink component: {component}"
        )
    try:
        physical = absolute.resolve(strict=True)
    except OSError as error:
        raise contracts.ContractError(f"{label} is missing: {absolute}") from error
    if any(component.is_symlink() for component in _path_components(physical)):
        raise contracts.ContractError(
            f"{label} physical path contains a symlink: {physical}"
        )
    return physical


def _stable_snapshot(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    physical = _physical_input_path(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(physical, flags)
    except OSError as error:
        raise contracts.ContractError(f"cannot open {label}: {physical}") from error
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(physical, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise contracts.ContractError(
                f"{label} must be one stable nonempty regular file"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(physical, follow_symlinks=False)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise contracts.ContractError(f"{label} changed while it was read")
        if any(
            getattr(after, field) != getattr(path_after, field)
            for field in stable_fields
        ):
            raise contracts.ContractError(f"{label} path identity changed while read")
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise contracts.ContractError(f"{label} read size changed")
        record = {
            "path": str(physical),
            "sha256": digest.hexdigest(),
            "size_bytes": after.st_size,
        }
        return physical, payload, record
    finally:
        os.close(descriptor)


def _authenticated_snapshot(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_size: int | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    expected_sha256 = _require_sha256(expected_sha256, f"{label} expected SHA-256")
    physical, payload, record = _stable_snapshot(path, label)
    if record["sha256"] != expected_sha256:
        raise contracts.ContractError(f"{label} changed from its expected SHA-256")
    if expected_size is not None and record["size_bytes"] != expected_size:
        raise contracts.ContractError(f"{label} changed from its expected size")
    return physical, payload, record


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = contracts.strict_json_loads(payload)
    except contracts.StrictJSONError as error:
        raise contracts.ContractError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise contracts.ContractError(f"{label} must be a JSON object")
    return value


def _descriptor(
    value: Any,
    review_root: Path,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or not isinstance(value["path"], str)
        or not value["path"]
        or isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] <= 0
    ):
        raise contracts.ContractError(f"{label} descriptor is invalid")
    expected_sha256 = _require_sha256(
        expected_sha256, f"{label} externally expected SHA-256"
    )
    descriptor_sha256 = _require_sha256(value["sha256"], f"{label} descriptor SHA-256")
    if expected_sha256 != descriptor_sha256:
        raise contracts.ContractError(
            f"{label} external SHA-256 does not match the review descriptor"
        )
    candidate = Path(value["path"])
    relative = not candidate.is_absolute()
    if relative:
        candidate = review_root / candidate
    physical, _payload, record = _authenticated_snapshot(
        candidate,
        descriptor_sha256,
        label,
        expected_size=value["size_bytes"],
    )
    if relative:
        try:
            physical.relative_to(review_root)
        except ValueError as error:
            raise contracts.ContractError(
                f"{label} escaped the derived-static review root"
            ) from error
    return physical, record


def _nested_descriptor(review: Mapping[str, Any], role: str) -> Any:
    if role not in ARTIFACT_PATHS:
        raise contracts.ContractError(f"unknown evidence role: {role}")
    value: Any = review
    for field in ARTIFACT_PATHS[role]:
        if not isinstance(value, Mapping) or field not in value:
            raise contracts.ContractError(
                f"derived-static review is missing evidence role {role}"
            )
        value = value[field]
    return value


def _validate_raw_static_decision(
    review: Mapping[str, Any], review_root: Path
) -> dict[str, Any]:
    raw = review["source_authorities"]["raw_static_decision"]
    _raw_path, raw_bytes, raw_record = _authenticated_snapshot(
        Path(raw["file"]["path"]),
        raw["file"]["sha256"],
        "preserved raw static decision",
        expected_size=raw["file"]["size_bytes"],
    )
    raw_payload = _json_object(raw_bytes, "preserved raw static decision")
    expected = {
        "rejected": ("rejected", "stop"),
        "approved_for_lod_and_binding": (
            "research_candidate",
            "lod_then_species_rig_binding",
        ),
    }.get(raw.get("decision"))
    if (
        expected is None
        or raw_payload.get("schema")
        != "avengine_controlled_animal_static_decision_v1"
        or raw_payload.get("decision") != raw.get("decision")
        or raw_payload.get("state_classification") != expected[0]
        or raw_payload.get("formal_dataset_registration_authorized") is not False
        or raw_payload.get("next_gate") != expected[1]
        or (
            "next_gate" in raw
            and raw.get("next_gate") != raw_payload.get("next_gate")
        )
        or (
            "checks" in raw
            and contracts.canonical_json(raw.get("checks"))
            != contracts.canonical_json(raw_payload.get("checks"))
        )
        or raw_payload.get("decision_sha256") != raw["decision_sha256"]
        or raw_payload.get("decision_sha256")
        != review_contract.hash_without(raw_payload, "decision_sha256")
    ):
        raise contracts.ContractError(
            "raw static decision no longer matches its frozen authority"
        )
    return {
        "decision": raw["decision"],
        "decision_sha256": raw["decision_sha256"],
        "file": raw_record,
        "formal_dataset_registration_authorized": False,
        "overwritten": False,
        "preserved": True,
        "state_classification": expected[0],
    }


def _validate_raw_rejection(
    review: Mapping[str, Any], review_root: Path
) -> dict[str, Any]:
    """Legacy v1 receipt helper restricted to an actual raw rejection."""

    result = _validate_raw_static_decision(review, review_root)
    if (
        result["decision"] != "rejected"
        or result["state_classification"] != "rejected"
    ):
        raise contracts.ContractError(
            "legacy raw static rejection helper requires a rejected authority"
        )
    return result


def _producer_record(path: Path, label: str) -> dict[str, Any]:
    _path, _payload, record = _stable_snapshot(path, label)
    return record


def _output_parent(path: Path) -> tuple[Path, str]:
    output = Path(path).absolute()
    if output.name in {"", ".", ".."} or output.suffix != ".json":
        raise contracts.ContractError("output must name one new .json receipt file")
    parent = output.parent
    for component in _path_components(parent):
        if component.is_symlink() and not _allowed_tmp_bridge(component):
            raise contracts.ContractError(
                f"output parent contains an unsafe symlink component: {component}"
            )
    try:
        physical_parent = parent.resolve(strict=True)
    except OSError as error:
        raise contracts.ContractError(
            f"output parent must already exist: {parent}"
        ) from error
    if not physical_parent.is_dir() or any(
        component.is_symlink() for component in _path_components(physical_parent)
    ):
        raise contracts.ContractError(
            f"output parent must be a direct physical directory: {physical_parent}"
        )
    return physical_parent, output.name


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while publishing rejection receipt")
        written += count


def _publish_atomic_no_replace(path: Path, payload: bytes) -> Path:
    parent, name = _output_parent(path)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory = os.open(parent, directory_flags)
    staging_name = f".{name}.{uuid.uuid4().hex}.staging"
    staging_created = False
    try:
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise contracts.ContractError(
                f"refusing to replace rejection receipt: {parent / name}"
            )
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(staging_name, file_flags, 0o600, dir_fd=directory)
        staging_created = True
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                staging_name,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise contracts.ContractError(
                f"rejection receipt appeared concurrently: {parent / name}"
            ) from error
        os.fsync(directory)
        os.unlink(staging_name, dir_fd=directory)
        staging_created = False
        os.fsync(directory)
    finally:
        if staging_created:
            try:
                os.unlink(staging_name, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)
    published = parent / name
    _path, published_bytes, _record = _stable_snapshot(
        published, "published rejection receipt"
    )
    if published_bytes != payload or stat.S_IMODE(published.stat().st_mode) != 0o444:
        raise contracts.ContractError(
            "published rejection receipt bytes or mode changed"
        )
    return published


def freeze_rejection(
    *,
    instance_id: str,
    review_path: Path,
    expected_review_file_sha256: str,
    expected_internal_review_sha256: str,
    observations: Sequence[Sequence[str]],
    output_path: Path,
) -> Path:
    expected_internal_review_sha256 = _require_sha256(
        expected_internal_review_sha256,
        "expected internal review SHA-256",
    )
    review_path, review_bytes, review_record = _authenticated_snapshot(
        review_path,
        expected_review_file_sha256,
        "derived-static review JSON",
    )
    review = review_contract.validate_review(
        _json_object(review_bytes, "derived-static review JSON")
    )
    if (
        review["instance_identity"]["instance_id"] != instance_id
        or review["review_sha256"] != expected_internal_review_sha256
    ):
        raise contracts.ContractError(
            "derived-static review instance or internal SHA-256 changed"
        )
    if (
        review["status"] != review_contract.REVIEW_STATUS
        or review["formal_dataset_registration_authorized"] is not False
        or review["human_review"]["status"] != "pending"
        or review["human_review"]["decision"] is not None
    ):
        raise contracts.ContractError(
            "machine rejection requires a still-pending derived-static review"
        )
    if (
        not isinstance(observations, Sequence)
        or isinstance(observations, (str, bytes))
        or not observations
    ):
        raise contracts.ContractError(
            "at least one enumerated objective defect observation is required"
        )
    review_root = review_path.parent
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_observation in enumerate(observations):
        if (
            not isinstance(raw_observation, Sequence)
            or isinstance(raw_observation, (str, bytes))
            or len(raw_observation) != 3
        ):
            raise contracts.ContractError(
                f"observation {index} must contain code, evidence role, and SHA-256"
            )
        code, role, expected_artifact_sha256 = raw_observation
        if code not in receipt_contract.DEFECT_CODES:
            raise contracts.ContractError(f"unknown objective defect code: {code}")
        if role not in receipt_contract.EVIDENCE_ROLES:
            raise contracts.ContractError(f"unknown evidence role: {role}")
        identity = (code, role)
        if identity in seen:
            raise contracts.ContractError(
                f"duplicate objective defect observation: {code}/{role}"
            )
        seen.add(identity)
        _artifact_path, artifact_record = _descriptor(
            _nested_descriptor(review, role),
            review_root,
            f"objective defect evidence {code}/{role}",
            expected_artifact_sha256,
        )
        records.append(
            {
                "artifact": artifact_record,
                "defect_code": code,
                "evidence_role": role,
                "expected_artifact_sha256": expected_artifact_sha256,
                "gate_result": "rejected",
                "observation_mode": receipt_contract.OBSERVATION_MODE,
            }
        )
    records.sort(key=lambda value: (value["defect_code"], value["evidence_role"]))
    raw_rejection = _validate_raw_rejection(review, review_root)
    receipt: dict[str, Any] = {
        "schema": receipt_contract.RECEIPT_SCHEMA,
        "status": receipt_contract.RECEIPT_STATUS,
        "scope": receipt_contract.RECEIPT_SCOPE,
        "instance_id": instance_id,
        "state_classification": "rejected",
        "formal_dataset_registration_authorized": False,
        "review_binding": {
            "review_file": review_record,
            "expected_external_review_file_sha256": expected_review_file_sha256,
            "internal_review_sha256": review["review_sha256"],
            "expected_internal_review_sha256": expected_internal_review_sha256,
            "review_status": review["status"],
            "human_review_status_at_freeze": review["human_review"]["status"],
            "human_review_decision_at_freeze": review["human_review"]["decision"],
        },
        "raw_static_rejection": raw_rejection,
        "observations": records,
        "authority": {
            "observer_kind": receipt_contract.OBSERVER_KIND,
            "user_decision_claimed": False,
            "human_decision_claimed": False,
            "approval_claimed": False,
            "product_definition_changed": False,
        },
        "effect": {
            "derived_static_admission": "rejected_fail_closed",
            "repaired_asset_registration_authorized": False,
            "source_asset_v2_authorized": False,
            "rigging_authorized": False,
            "animation_authorized": False,
            "ue_execution_authorized": False,
            "native_change_authorized": False,
            "next_gate": receipt_contract.NEXT_GATE,
        },
        "automatic_checks": {
            "external_review_file_sha256_matched": True,
            "internal_review_sha256_matched": True,
            "review_contract_revalidated": True,
            "raw_static_rejection_preserved": True,
            "all_defect_codes_enumerated": True,
            "all_evidence_artifacts_reauthenticated": True,
            "no_user_or_human_decision_claimed": True,
            "no_approval_or_registration_authorized": True,
            "no_ue_or_native_execution_performed": True,
            "overall": True,
        },
        "producer": {
            "tool": _producer_record(Path(__file__), "machine rejection producer"),
            "contract": _producer_record(
                Path(receipt_contract.__file__),
                "machine rejection contract",
            ),
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = receipt_contract.hash_without(receipt, "receipt_sha256")
    receipt = receipt_contract.validate_receipt(receipt)
    encoded = (contracts.canonical_json(receipt) + "\n").encode("utf-8")
    return _publish_atomic_no_replace(output_path, encoded)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--expected-review-file-sha256", required=True)
    parser.add_argument("--expected-internal-review-sha256", required=True)
    parser.add_argument(
        "--observation",
        required=True,
        action="append",
        nargs=3,
        metavar=("DEFECT_CODE", "EVIDENCE_ROLE", "EXPECTED_ARTIFACT_SHA256"),
        help=(
            "Repeat for each objective defect. Codes and roles are strict enums; "
            "the artifact SHA must match both the review descriptor and file."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output = freeze_rejection(
            instance_id=args.instance_id,
            review_path=args.review,
            expected_review_file_sha256=args.expected_review_file_sha256,
            expected_internal_review_sha256=args.expected_internal_review_sha256,
            observations=args.observation,
            output_path=args.output,
        )
        _path, payload, record = _stable_snapshot(output, "published rejection receipt")
        receipt = receipt_contract.validate_receipt(
            _json_object(payload, "published rejection receipt")
        )
    except (
        contracts.ContractError,
        receipt_contract.MachineRejectionContractError,
        review_contract.DerivedStaticReviewContractError,
        OSError,
    ) as error:
        print(
            f"CONTROLLED_ANIMAL_DERIVED_STATIC_MACHINE_REJECTION_FAILED {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "CONTROLLED_ANIMAL_DERIVED_STATIC_MACHINE_REJECTION_OK "
        f"instance={receipt['instance_id']} "
        f"receipt_sha256={receipt['receipt_sha256']} "
        f"file_sha256={record['sha256']} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
