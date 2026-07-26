#!/usr/bin/env python3
"""Collect historical human review decisions into a VLM calibration manifest.

Before any visual-model pre-screener may triage animal review media, it must
be scored against the retained human decisions on identical inputs.  This
tool walks the given search roots, finds authenticated human decision records
and pairs them with their review media, producing one deterministic
calibration manifest.  It never calls a model and never mutates the sources;
it only fixes the benchmark a candidate pre-screener has to match.

Collected record families:
- ``animation_decision.json`` documents
  (schema ``avengine_controlled_animal_animation_decision_v1``): the six
  boolean human checks (walking_direction, deformation, ground contact,
  stability, detached geometry) plus the approved/rejected conclusion.
- legacy canonical-image ``review_decision.json`` documents: the 2D
  anatomy/breed hard-gate decisions taken before Pixel3D.

The manifest deliberately keeps records whose media files are missing
(``media_present: false``) so coverage loss is visible instead of silently
shrinking the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "avengine_vlm_review_calibration_set_v1"
ANIMATION_DECISION_SCHEMA = "avengine_controlled_animal_animation_decision_v1"
ANIMATION_CHECK_FIELDS = (
    "walking_direction",
    "walking_limb_deformation",
    "walking_ground_contact",
    "idle_ground_contact",
    "body_stability",
    "detached_geometry_absent",
)
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".mp4", ".webm")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        required=True,
        help="Directory tree to scan; may be given multiple times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def media_record(path: Path) -> dict:
    if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "media_present": True,
        }
    else:
        return {"path": str(path), "media_present": False}


def sibling_media(directory: Path) -> list[dict]:
    records = []
    for candidate in sorted(directory.iterdir()):
        if candidate.suffix.lower() in MEDIA_SUFFIXES and candidate.is_file():
            records.append(media_record(candidate))
    return records


def collect_animation_decision(path: Path, payload: dict) -> dict | None:
    if payload.get("schema") != ANIMATION_DECISION_SCHEMA:
        return None
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    media = []
    review = payload.get("review")
    if isinstance(review, dict) and review.get("path"):
        review_path = Path(review["path"])
        media.append(media_record(review_path))
        if review_path.parent.is_dir():
            media.extend(sibling_media(review_path.parent))
    return {
        "category": "animation_six_checks",
        "decision_file": {"path": str(path), "sha256": sha256_file(path)},
        "asset_id": payload.get("asset_id"),
        "human_verdict": {
            "decision": payload.get("decision"),
            "checks": {
                field: checks.get(field) for field in ANIMATION_CHECK_FIELDS
            },
        },
        "media": media,
    }


def collect_canonical_2d_decision(path: Path, payload: dict) -> dict | None:
    if "decision" not in payload or "pixel3d_authorized" not in payload:
        return None
    return {
        "category": "canonical_2d",
        "decision_file": {"path": str(path), "sha256": sha256_file(path)},
        "asset_id": path.parent.name,
        "human_verdict": {
            key: value
            for key, value in payload.items()
            if key not in ("schema",)
        },
        "media": sibling_media(path.parent),
    }


def collect_records(search_roots: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for root in search_roots:
        root = root.resolve()
        if not root.is_dir():
            raise SystemExit(f"search root is not a directory: {root}")
        for name in ("animation_decision.json", "review_decision.json"):
            for path in sorted(root.rglob(name)):
                path = path.resolve()
                if path in seen or path.is_symlink() or not path.is_file():
                    continue
                seen.add(path)
                payload = load_json(path)
                if not isinstance(payload, dict):
                    continue
                if name == "animation_decision.json":
                    record = collect_animation_decision(path, payload)
                else:
                    record = collect_canonical_2d_decision(path, payload)
                if record is not None:
                    records.append(record)
    records.sort(key=lambda item: item["decision_file"]["path"])
    return records


def summarize(records: list[dict]) -> dict:
    by_category = {}
    for record in records:
        category = record["category"]
        bucket = by_category.setdefault(
            category, {"count": 0, "with_media": 0}
        )
        bucket["count"] += 1
        if any(item.get("media_present") for item in record["media"]):
            bucket["with_media"] += 1
    return by_category


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace calibration manifest: {output}")
    records = collect_records(args.search_root)
    if not records:
        raise SystemExit("no human review decisions found under the search roots")
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "fixed human-decision benchmark for scoring a candidate visual "
            "pre-screener; the pre-screener is triage support and never "
            "review authority"
        ),
        "search_roots": [str(Path(root).resolve()) for root in args.search_root],
        "record_count": len(records),
        "summary_by_category": summarize(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "VLM_REVIEW_CALIBRATION_SET_OK "
        f"records={len(records)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
