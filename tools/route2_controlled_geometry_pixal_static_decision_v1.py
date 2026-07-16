#!/usr/bin/env python3
"""Publish immutable agent decisions for controlled-geometry Pixal reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "route2_controlled_geometry_pixal_static_agent_qa_v1"
REVIEW_SCHEMA = "route2_controlled_geometry_pixal_static_review_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
PIXAL_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
ALLOWED_STATUS = ("agent_static_visual_passed", "rejected")
REQUIRED_ARTIFACTS = {
    "front.png",
    "back.png",
    "side.png",
    "top.png",
    "quarter.png",
    "contact_sheet.png",
}


class DecisionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, require_mode: int | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size <= 0
    ):
        raise DecisionError(f"artifact must be a direct nonempty file: {path}")
    if require_mode is not None and stat.S_IMODE(path.stat().st_mode) != require_mode:
        raise DecisionError(f"artifact mode changed: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def authenticate_review(asset_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    review_root = PIXAL_ROOT / asset_id / "static_review_v1"
    manifest_path = review_root / "review_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionError(f"static review manifest is invalid: {error}") from error
    artifacts = manifest.get("artifacts")
    checks = manifest.get("automatic_checks")
    if (
        manifest.get("schema") != REVIEW_SCHEMA
        or manifest.get("asset_id") != asset_id
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("front_axis") != "positive-y"
        or manifest.get("up_axis") != "positive-z"
        or manifest.get("agent_visual_qa") != "pending"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != REQUIRED_ARTIFACTS
    ):
        raise DecisionError("static review schema, axes, automatic checks, or state changed")
    for filename in REQUIRED_ARTIFACTS:
        if artifacts[filename] != record(review_root / filename, require_mode=0o444):
            raise DecisionError(f"static review artifact changed: {filename}")
    return review_root, manifest, record(manifest_path, require_mode=0o444)


def publish(asset_id: str, status: str, notes: str) -> Path:
    if status not in ALLOWED_STATUS:
        raise DecisionError(f"status must be one of {ALLOWED_STATUS}")
    if not notes.strip():
        raise DecisionError("notes must be nonempty")
    review_root, manifest, manifest_record = authenticate_review(asset_id)
    destination = review_root / "agent_static_visual_qa.json"
    passed = status == "agent_static_visual_passed"
    payload = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "status": status,
        "state_classification": "research_candidate" if passed else "rejected",
        "reviewer_kind": "agent",
        "reviewer": "codex_female_route2_base",
        "review_manifest": manifest_record,
        "contact_sheet": manifest["artifacts"]["contact_sheet.png"],
        "checks": {
            "target_geometry_matches_approved_2d": passed,
            "identity_pose_and_non_target_geometry_preserved": passed,
            "front_back_side_top_consistent": passed,
            "no_obvious_missing_limb_hole_or_fusion": passed,
            "pbr_material_and_images_present": passed,
            "positive_y_front_and_positive_z_up_plausible": passed,
            "bindable_static_silhouette_plausible": passed,
        },
        "notes": notes.strip(),
        "tokenrig_preflight_authorized": passed,
        "formal_dataset_registration_authorized": False,
        "user_acceptance": "not_claimed",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o444,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--status", choices=ALLOWED_STATUS, required=True)
    parser.add_argument("--notes", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"CONTROLLED_PIXAL_STATIC_DECISION_OK {publish(args.asset_id, args.status, args.notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
