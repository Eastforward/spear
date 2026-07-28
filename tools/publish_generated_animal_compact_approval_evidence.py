#!/usr/bin/env python3
"""Publish the minimal reusable-style and current-asset approval receipts.

The motion-style receipt records an explicit approval of the Idle/Walking
style shown by one existing video.  The current-asset receipt binds an exact
v4 animation review to its reviewed GLB and Idle/Walking side readbacks.
Neither command performs codec inspection or repeats six-view review.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


MOTION_STYLE_APPROVAL_SCHEMA = (
    "avengine_generated_animal_motion_style_approval_v1"
)
CURRENT_ASSET_SHORT_READBACK_SCHEMA = (
    "avengine_generated_animal_current_asset_short_readback_v1"
)
EXACT_V4_REVIEW_SCHEMA = (
    "avengine_target_native_generated_quadruped_review_run_v4"
)
ANIMATION_ACTIONS = ["Idle", "Walking"]
CURRENT_ASSET_SHORT_READBACK_CHECKS = {
    "current_review_action_media_bound": True,
    "reviewed_animated_glb_bound": True,
    "idle_action_present": True,
    "walking_action_present": True,
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class EvidenceError(contracts.ContractError):
    """Raised when compact approval evidence is incomplete or has changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_nonempty_file(path: Path, label: str) -> Path:
    literal = Path(path)
    if literal.is_symlink():
        raise EvidenceError(f"{label} must be a direct file, not a symlink")
    try:
        resolved = literal.resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"{label} is missing: {literal}") from error
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size <= 0
    ):
        raise EvidenceError(f"{label} is missing, empty, or unsafe: {resolved}")
    return resolved


def _absolute_record(path: Path, label: str) -> dict[str, Any]:
    path = _direct_nonempty_file(path, label)
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EvidenceError(f"{label} changed while it was being authenticated")
    return {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": after.st_size,
    }


def _verified_descriptor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} descriptor is missing")
    required = {"path", "sha256", "size_bytes"}
    if not required.issubset(value):
        raise EvidenceError(f"{label} descriptor fields are incomplete")
    raw_path = value.get("path")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(sha256, str)
        or SHA256_PATTERN.fullmatch(sha256) is None
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise EvidenceError(f"{label} descriptor values are invalid")
    expected = {
        "path": raw_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    actual = _absolute_record(Path(raw_path), label)
    if actual != expected:
        raise EvidenceError(
            f"{label} changed or its descriptor is non-canonical"
        )
    return actual


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = {name: item for name, item in value.items() if name != key}
    return hashlib.sha256(
        contracts.canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _require_asset_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or ASSET_ID_PATTERN.fullmatch(value) is None
    ):
        raise EvidenceError(
            f"{label} must be a lowercase controlled-asset identifier"
        )
    return value


def _review_asset_id(review: Mapping[str, Any]) -> str:
    inputs = review.get("inputs")
    target_lineage = (
        inputs.get("target_rig_lineage")
        if isinstance(inputs, Mapping)
        else None
    )
    if not isinstance(target_lineage, Mapping):
        raise EvidenceError(
            "exact v4 review inputs.target_rig_lineage is missing"
        )
    observed = _require_asset_id(
        target_lineage.get("asset_id"),
        "exact v4 review target-rig asset_id",
    )

    forward_contract = review.get("forward_contract")
    if isinstance(forward_contract, Mapping):
        forward_lineage = forward_contract.get("target_rig_lineage")
        if forward_lineage is not None:
            if not isinstance(forward_lineage, Mapping):
                raise EvidenceError(
                    "exact v4 review forward target-rig lineage is invalid"
                )
            forward_asset_id = _require_asset_id(
                forward_lineage.get("asset_id"),
                "exact v4 review forward target-rig asset_id",
            )
            if forward_asset_id != observed:
                raise EvidenceError(
                    "exact v4 review target-rig asset identity is inconsistent"
                )
    return observed


def _load_exact_v4_review(
    path: Path,
    *,
    expected_asset_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    review_path = _direct_nonempty_file(path, "exact v4 animation review")
    before = _absolute_record(review_path, "exact v4 animation review")
    payload = contracts.load_json(review_path)
    after = _absolute_record(review_path, "exact v4 animation review")
    if before != after:
        raise EvidenceError(
            "exact v4 animation review changed while it was being authenticated"
        )
    if not isinstance(payload, Mapping):
        raise EvidenceError("exact v4 animation review root must be an object")
    if payload.get("schema") != EXACT_V4_REVIEW_SCHEMA:
        raise EvidenceError("animation review is not the exact v4 schema")
    if (
        payload.get("status")
        != "research_candidate_pending_human_review"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise EvidenceError(
            "exact v4 animation review state is not eligible for compact approval"
        )
    automatic_gates = payload.get("automatic_admission_gates")
    if (
        not isinstance(automatic_gates, Mapping)
        or automatic_gates.get("all_automatic_gates_passed") is not True
    ):
        raise EvidenceError(
            "exact v4 animation review automatic gates did not pass"
        )
    observed_asset_id = _review_asset_id(payload)
    if observed_asset_id != expected_asset_id:
        raise EvidenceError(
            "exact v4 animation review asset_id does not match --asset-id"
        )
    return dict(payload), after


def publish_motion_style_approval(
    *,
    style_video: Path,
    output: Path,
) -> Path:
    """Publish a reusable Idle/Walking style approval for one video."""

    payload: dict[str, Any] = {
        "schema": MOTION_STYLE_APPROVAL_SCHEMA,
        "status": "approved_for_idle_walking_motion_style",
        "actions": list(ANIMATION_ACTIONS),
        "evidence_video": _absolute_record(
            style_video,
            "motion-style evidence video",
        ),
    }
    payload["approval_sha256"] = _hash_without(
        payload,
        "approval_sha256",
    )
    return contracts.write_json_no_replace(output, payload)


def publish_current_asset_short_readback(
    *,
    review: Path,
    asset_id: str,
    output: Path,
) -> Path:
    """Publish the exact current GLB and Idle/Walking readback binding."""

    asset_id = _require_asset_id(asset_id, "--asset-id")
    review_payload, review_record = _load_exact_v4_review(
        review,
        expected_asset_id=asset_id,
    )
    outputs = review_payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EvidenceError("exact v4 animation review outputs are missing")
    animated_glb = _verified_descriptor(
        outputs.get("animated_glb"),
        "exact v4 reviewed animated GLB",
    )
    media = outputs.get("media")
    if not isinstance(media, Mapping):
        raise EvidenceError("exact v4 animation review media are missing")
    action_readbacks = {
        "Idle": _verified_descriptor(
            media.get("idle_side"),
            "exact v4 idle_side readback",
        ),
        "Walking": _verified_descriptor(
            media.get("walking_side"),
            "exact v4 walking_side readback",
        ),
    }

    payload: dict[str, Any] = {
        "schema": CURRENT_ASSET_SHORT_READBACK_SCHEMA,
        "status": "passed_current_asset_geometry_and_actions",
        "asset_id": asset_id,
        "animation_review": review_record,
        "reviewed_animated_glb": animated_glb,
        "actions": list(ANIMATION_ACTIONS),
        "action_readbacks": action_readbacks,
        "checks": dict(CURRENT_ASSET_SHORT_READBACK_CHECKS),
    }
    payload["receipt_sha256"] = _hash_without(
        payload,
        "receipt_sha256",
    )
    return contracts.write_json_no_replace(output, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish minimal generated-animal motion-style or current-asset "
            "approval evidence without replacing existing artifacts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    motion_style = subparsers.add_parser(
        "motion-style",
        help="approve reusable Idle/Walking motion style shown by one video",
    )
    motion_style.add_argument("--style-video", type=Path, required=True)
    motion_style.add_argument("--output", type=Path, required=True)

    current_readback = subparsers.add_parser(
        "current-readback",
        help="bind one exact v4 review to its current GLB and side readbacks",
    )
    current_readback.add_argument("--review", type=Path, required=True)
    current_readback.add_argument("--asset-id", required=True)
    current_readback.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "motion-style":
            published = publish_motion_style_approval(
                style_video=args.style_video,
                output=args.output,
            )
        else:
            published = publish_current_asset_short_readback(
                review=args.review,
                asset_id=args.asset_id,
                output=args.output,
            )
    except (contracts.ContractError, OSError) as error:
        print(
            f"GENERATED_ANIMAL_COMPACT_EVIDENCE_FAILED: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "GENERATED_ANIMAL_COMPACT_EVIDENCE_OK "
        f"kind={args.command} output={published.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
