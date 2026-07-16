#!/usr/bin/env python3

"""Build the authenticated inventory for the 115 native Rocketbox avatars.

The pure-Python half discovers only each avatar folder's canonical non-facial
FBX.  Blender measurements are generated separately and merged by both the
path-unique ``base_avatar_id`` and the exact FBX SHA-256.  Authored stature is
never normalized: the Apartment contract always uses Unreal scale ``1.0``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


ROCKETBOX_COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
SCHEMA_VERSION = "rocketbox_human_inventory_v1"
AUDIT_SCHEMA = "rocketbox_blender_audit_shard_v1"
CATEGORIES = ("Adults", "Children", "Professions")
EXPECTED_COUNTS = {
    "total": 115,
    "gender": {"male": 74, "female": 41},
    "category": {"Adults": 40, "Children": 4, "Professions": 71},
}
HEIGHT_RANGES_CM = {
    "Adults": (140.0, 215.0),
    "Professions": (140.0, 215.0),
    "Children": (80.0, 170.0),
}
MINIMUM_CEILING_HEADROOM_CM = 25.0
DEFAULT_APARTMENT_CEILING_CM = 280.0
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _gender(folder_name: str) -> str:
    lowered = folder_name.lower()
    has_female = "female" in lowered
    has_male = "male" in lowered and not has_female
    if has_female == has_male:
        raise RuntimeError(f"cannot authenticate gender from folder name: {folder_name}")
    return "female" if has_female else "male"


def _file_record(path: Path, root: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "relative_path": path.relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "git_blob_sha1": git_blob_sha1_file(path),
    }


def discover_canonical_avatars(rocketbox_root: Path) -> list[dict]:
    """Return deterministic path-unique records for canonical non-facial FBXs."""

    root = Path(rocketbox_root).resolve()
    avatars_root = root / "Assets" / "Avatars"
    if not avatars_root.is_dir():
        raise RuntimeError(f"Rocketbox avatar root is missing: {avatars_root}")
    records = []
    for category in CATEGORIES:
        category_dir = avatars_root / category
        if not category_dir.is_dir():
            raise RuntimeError(f"Rocketbox category is missing: {category_dir}")
        for avatar_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            canonical_fbx = avatar_dir / "Export" / f"{avatar_dir.name}.fbx"
            # The Professions tree contains two historical facial-only party
            # aliases.  They are not additional base avatars.
            if not canonical_fbx.is_file():
                facial = avatar_dir / "Export" / f"{avatar_dir.name}_facial.fbx"
                if facial.is_file():
                    continue
                raise RuntimeError(f"canonical avatar FBX is missing: {canonical_fbx}")
            texture_dir = avatar_dir / "Textures"
            if not texture_dir.is_dir():
                raise RuntimeError(f"avatar texture directory is missing: {texture_dir}")
            texture_paths = sorted(path for path in texture_dir.iterdir() if path.is_file())
            if not texture_paths:
                raise RuntimeError(f"avatar has no texture/source files: {texture_dir}")
            preview = avatar_dir / f"{avatar_dir.name}.png"
            if not preview.is_file():
                raise RuntimeError(f"avatar preview is missing: {preview}")
            base_avatar_id = f"rocketbox_{_slug(category)}_{_slug(avatar_dir.name)}"
            gender = _gender(avatar_dir.name)
            fbx_record = _file_record(canonical_fbx, root)
            records.append(
                {
                    "base_avatar_id": base_avatar_id,
                    "legacy_asset_id": f"rocketbox_{_slug(avatar_dir.name)}",
                    "source_folder_name": avatar_dir.name,
                    "category": category,
                    "demographic": "child" if category == "Children" else "adult",
                    "gender": gender,
                    "front_axis": "-Y",
                    "up_axis": "+Z",
                    "fbx_path": str(canonical_fbx.resolve()),
                    "fbx_relative_path": fbx_record["relative_path"],
                    "fbx_size_bytes": fbx_record["size_bytes"],
                    "fbx_sha256": fbx_record["sha256"],
                    "fbx_git_blob_sha1": fbx_record["git_blob_sha1"],
                    "texture_dir": str(texture_dir.resolve()),
                    "source_files": {
                        "preview": _file_record(preview, root),
                        "textures": [_file_record(path, root) for path in texture_paths],
                    },
                }
            )
    ids = [record["base_avatar_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("path-unique Rocketbox base_avatar_id collision")
    return records


def demographic_height_contract(
    *, category: str, authored_height_cm: float, apartment_ceiling_cm: float
) -> dict:
    """Evaluate stature without changing the source actor's scale."""

    if category not in HEIGHT_RANGES_CM:
        raise RuntimeError(f"unsupported Rocketbox category: {category}")
    height = float(authored_height_cm)
    ceiling = float(apartment_ceiling_cm)
    if not math.isfinite(height) or height <= 0.0:
        raise RuntimeError("authored height must be a positive finite value")
    if not math.isfinite(ceiling) or ceiling <= 0.0:
        raise RuntimeError("apartment ceiling must be a positive finite value")
    minimum, maximum = HEIGHT_RANGES_CM[category]
    headroom = ceiling - height
    errors = []
    if height < minimum or height > maximum:
        errors.append(
            f"authored height {height:.3f}cm is outside {category} range "
            f"[{minimum:.1f}, {maximum:.1f}]cm"
        )
    if headroom < MINIMUM_CEILING_HEADROOM_CM:
        errors.append(
            f"apartment ceiling headroom {headroom:.3f}cm is below "
            f"{MINIMUM_CEILING_HEADROOM_CM:.1f}cm"
        )
    # A face/mouth proxy is better than the old fixed adult 155cm audio
    # source.  It scales naturally for children while remaining below the top
    # of head and is recorded for later head-bone refinement in UE.
    mouth_height = height * (0.90 if category != "Children" else 0.88)
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "category": category,
        "authored_height_cm": height,
        "allowed_height_cm": [minimum, maximum],
        "authored_height_preserved": True,
        "actor_scale": 1.0,
        "apartment_ceiling_cm": ceiling,
        "ceiling_headroom_cm": headroom,
        "minimum_ceiling_headroom_cm": MINIMUM_CEILING_HEADROOM_CM,
        "mouth_audio_height_cm": mouth_height,
        "audio_height_source": "0.90x authored height; 0.88x for children",
    }


def merge_blender_audits(
    inventory_records: list[dict],
    audit_paths: Iterable[Path],
    apartment_ceiling_cm: float = DEFAULT_APARTMENT_CEILING_CM,
) -> list[dict]:
    audits: dict[str, dict] = {}
    for path in audit_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != AUDIT_SCHEMA:
            raise RuntimeError(f"unexpected Blender audit schema: {path}")
        for audit in payload.get("records", []):
            avatar_id = audit.get("base_avatar_id")
            if avatar_id in audits:
                raise RuntimeError(f"duplicate Blender audit: {avatar_id}")
            audits[avatar_id] = audit
    merged = []
    for source in inventory_records:
        avatar_id = source["base_avatar_id"]
        audit = audits.get(avatar_id)
        if audit is None:
            raise RuntimeError(f"missing Blender audit: {avatar_id}")
        if audit.get("fbx_sha256") != source["fbx_sha256"]:
            raise RuntimeError(f"Blender audit FBX hash mismatch: {avatar_id}")
        if audit.get("status") != "passed":
            raise RuntimeError(f"Blender audit failed: {avatar_id}: {audit.get('errors')}")
        contract = demographic_height_contract(
            category=source["category"],
            authored_height_cm=audit["authored_height_cm"],
            apartment_ceiling_cm=apartment_ceiling_cm,
        )
        record = dict(source)
        record["blender_audit"] = audit
        record["height_contract"] = contract
        record["inventory_status"] = (
            "passed" if contract["status"] == "passed" else "failed"
        )
        merged.append(record)
    extra = sorted(set(audits) - {r["base_avatar_id"] for r in inventory_records})
    if extra:
        raise RuntimeError(f"Blender audit contains unknown avatars: {extra}")
    return merged


def validate_expected_population(records: list[dict]) -> dict:
    totals = {
        "total": len(records),
        "gender": dict(sorted(Counter(r["gender"] for r in records).items())),
        "category": dict(sorted(Counter(r["category"] for r in records).items())),
    }
    if totals != {
        "total": EXPECTED_COUNTS["total"],
        "gender": dict(sorted(EXPECTED_COUNTS["gender"].items())),
        "category": dict(sorted(EXPECTED_COUNTS["category"].items())),
    }:
        raise RuntimeError(f"Rocketbox population totals changed: {totals}")
    return totals


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_tsv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "base_avatar_id",
        "legacy_asset_id",
        "category",
        "demographic",
        "gender",
        "authored_height_cm",
        "height_status",
        "fbx_relative_path",
        "fbx_sha256",
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "base_avatar_id": record["base_avatar_id"],
                    "legacy_asset_id": record["legacy_asset_id"],
                    "category": record["category"],
                    "demographic": record["demographic"],
                    "gender": record["gender"],
                    "authored_height_cm": record["blender_audit"]["authored_height_cm"],
                    "height_status": record["height_contract"]["status"],
                    "fbx_relative_path": record["fbx_relative_path"],
                    "fbx_sha256": record["fbx_sha256"],
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocketbox-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument(
        "--apartment-ceiling-cm", type=float, default=DEFAULT_APARTMENT_CEILING_CM
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.rocketbox_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if commit != ROCKETBOX_COMMIT:
        raise RuntimeError(f"Rocketbox commit changed: {commit}")
    records = discover_canonical_avatars(root)
    totals = validate_expected_population(records)
    merged = merge_blender_audits(records, args.audit, args.apartment_ceiling_cm)
    failed = [r["base_avatar_id"] for r in merged if r["inventory_status"] != "passed"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "Microsoft-Rocketbox",
        "checkout_commit": commit,
        "license_spdx": "MIT",
        "front_axis": "-Y",
        "up_axis": "+Z",
        "population": totals,
        "apartment_height_policy": {
            "actor_scale": 1.0,
            "authored_height_preserved": True,
            "ceiling_cm": args.apartment_ceiling_cm,
            "allowed_ranges_cm": HEIGHT_RANGES_CM,
            "minimum_headroom_cm": MINIMUM_CEILING_HEADROOM_CM,
        },
        "automatic_checks": {
            "overall": "passed" if not failed else "failed",
            "failed_avatar_ids": failed,
        },
        "avatars": merged,
    }
    _atomic_json(args.output_json, payload)
    _write_tsv(args.output_tsv, merged)
    if failed:
        raise RuntimeError(f"height/room contract failed for {len(failed)} avatars")
    print(
        f"ROCKETBOX_INVENTORY_OK total={len(merged)} "
        f"output={args.output_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
