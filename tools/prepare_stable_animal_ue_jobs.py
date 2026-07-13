#!/usr/bin/env python3
"""Prepare authenticated, non-overwriting UE import jobs for stable animals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


SELECTION_SCHEMA = "avengine_stable_animal_ue_selection_v1"
REGISTRY_SCHEMA = "avengine_quaternius_stable_template_registry_v1"
BATCH_SCHEMA = "stable_animal_ue_import_batch_v1"
PREPARATION_SCHEMA = "stable_animal_ue_import_preparation_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe artifact: {path}")
    return {
        "path": str((published_path or path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _record_matches(record: dict[str, Any]) -> bool:
    try:
        path = Path(record["path"]).resolve()
        return bool(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and _sha256(path) == record["sha256"]
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not token:
        raise ValueError(f"unsafe empty tag token from {value!r}")
    return token


def _validate_entry(entry: dict[str, Any]) -> None:
    if (
        entry.get("state_classification") != "research_candidate"
        or entry.get("formal_dataset_registration_authorized") is not False
        or set(entry.get("actions", [])) != {"Walking", "Idle"}
        or entry.get("direction", {}).get("automatic_fine_yaw_inference") is not False
        or entry.get("direction", {}).get("review_status")
        != "agent_selected_pending_human_review"
        or entry.get("qa", {}).get("ue_apartment_media") != "pending"
        or entry.get("qa", {}).get("human_visual_review") != "pending"
        or not str(entry.get("qa", {}).get("walking_deformation", "")).startswith(
            "passed_"
        )
        or not str(entry.get("qa", {}).get("idle_deformation", "")).startswith(
            "passed_"
        )
        or not _record_matches(entry.get("runtime_glb", {}))
        or not _record_matches(entry.get("deformation_audit", {}))
    ):
        raise ValueError(f"stable template entry is not UE-canary ready: {entry.get('template_id')}")


def build_jobs(registry: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("stable template registry schema changed")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("stable UE selection schema changed")
    by_id = {entry["template_id"]: entry for entry in registry.get("entries", [])}
    selected = selection.get("selections")
    if not isinstance(selected, list) or not selected:
        raise ValueError("stable UE selection must contain at least one template")
    jobs = []
    seen_ids: set[str] = set()
    seen_tags: set[str] = set()
    for item in selected:
        template_id = str(item.get("template_id", ""))
        if template_id in seen_ids or template_id not in by_id:
            raise ValueError(f"unknown or duplicate stable template: {template_id}")
        entry = by_id[template_id]
        _validate_entry(entry)
        species = _safe_token(str(item.get("species", "")))
        breed = _safe_token(str(item.get("breed", "")))
        actor_scale = float(item.get("actor_scale", 0.0))
        audio_height = float(item.get("audio_source_height_offset_m", 0.0))
        audio_lookup = str(item.get("audio_lookup", ""))
        if (
            not math.isfinite(actor_scale)
            or not 0.01 <= actor_scale <= 1.0
            or not math.isfinite(audio_height)
            or not 0.05 <= audio_height <= 3.0
            or not audio_lookup
        ):
            raise ValueError(f"invalid stable Apartment profile: {template_id}")
        tag = f"stable_{species}_{breed}_{_safe_token(template_id)}"
        if tag in seen_tags:
            raise ValueError(f"duplicate stable UE tag: {tag}")
        jobs.append(
            {
                "asset_id": template_id,
                "template_id": template_id,
                "tag": tag,
                "taxonomy_label": entry["taxonomy_label"],
                "species": species,
                "breed": breed,
                "actor_scale": actor_scale,
                "audio_lookup": audio_lookup,
                "audio_source_height_offset_m": audio_height,
                "walking_forward_yaw_offset_deg": float(
                    entry["direction"]["cardinal_yaw_deg"]
                ),
                "rigged_glb": entry["runtime_glb"]["path"],
                "rigged_glb_sha256": entry["runtime_glb"]["sha256"],
                "expected_actions": ["Idle", "Walking"],
                "deformation_audit": entry["deformation_audit"],
                "human_review_status": entry["direction"]["review_status"],
                "formal_dataset_registration_authorized": False,
            }
        )
        seen_ids.add(template_id)
        seen_tags.add(tag)
    return sorted(jobs, key=lambda item: item["template_id"])


def prepare(*, registry_path: Path, selection_path: Path, output_root: Path) -> Path:
    registry_path = registry_path.resolve()
    selection_path = selection_path.resolve()
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"refusing to replace output: {output_root}")
    registry = _load_json(registry_path)
    selection = _load_json(selection_path)
    jobs = build_jobs(registry, selection)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        jobs_payload = {
            "schema": BATCH_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "non_destructive_policy": (
                "unique stable_* gate content directories; preserve Pixal, Hunyuan, "
                "Rocketbox, and all earlier stable revisions"
            ),
            "registry": _artifact(registry_path),
            "selection": _artifact(selection_path),
            "job_count": len(jobs),
            "jobs": jobs,
        }
        jobs_path = staging / "ue_import_jobs.json"
        jobs_path.write_text(
            json.dumps(jobs_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published_jobs = output_root / jobs_path.name
        preparation = {
            "schema": PREPARATION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "jobs": _artifact(jobs_path, published_path=published_jobs),
            "job_count": len(jobs),
            "automatic_checks": {
                "registry_reauthenticated": True,
                "runtime_glbs_reauthenticated": True,
                "walk_idle_deformation_passed": True,
                "fine_yaw_inference_disabled": True,
                "human_review_still_pending": True,
                "formal_registration_authorized": False,
            },
        }
        (staging / "ue_import_preparation_manifest.json").write_text(
            json.dumps(preparation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return output_root / "ue_import_jobs.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    jobs_path = prepare(
        registry_path=args.registry,
        selection_path=args.selection,
        output_root=args.output_root,
    )
    print(f"STABLE_ANIMAL_UE_JOBS_OK jobs={jobs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
