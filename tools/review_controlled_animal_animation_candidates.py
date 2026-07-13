#!/usr/bin/env python3
"""Authenticate and freeze Walk/Idle visual decisions for controlled animals."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_animation_reviews as animation_reviews


DECISIONS_SCHEMA = "avengine_controlled_animal_animation_decisions_v1"
DECISION_SCHEMA = "avengine_controlled_animal_animation_decision_v1"
DECISION_BATCH_SCHEMA = "avengine_controlled_animal_animation_decision_batch_v1"
CHECK_FIELDS = {
    "walking_direction",
    "walking_limb_deformation",
    "walking_ground_contact",
    "idle_ground_contact",
    "body_stability",
    "detached_geometry_absent",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_relative_artifact(
    root: Path, artifact: Mapping[str, Any], *, label: str
) -> Path:
    path = (root / str(artifact.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped review root") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != artifact.get("size_bytes")
        or _sha256_file(path) != artifact.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def load_review_batch(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"animation review batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != animation_reviews.BATCH_SCHEMA
        or payload.get("status") != "rendered_pending_visual_qa"
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("review_count") != len(payload.get("reviews", []))
        or payload.get("passed_render_count") != payload.get("review_count")
        or payload.get("failed_render_count") != 0
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("animation review batch contract/hash is invalid")
    root = path.parent
    reviews: dict[str, Any] = {}
    for index in payload["reviews"]:
        review_path = _verify_relative_artifact(
            root, index["review"], label="animation review manifest"
        )
        review = contracts.load_json(review_path)
        asset_id = index.get("asset_id")
        if (
            asset_id in reviews
            or review.get("schema") != animation_reviews.REVIEW_SCHEMA
            or review.get("asset_id") != asset_id
            or review.get("review_sha256") != index.get("review_sha256")
            or review.get("review_sha256") != _hash_without(review, "review_sha256")
            or review.get("automatic_checks", {}).get("overall") != "passed"
            or set(review.get("videos", {}))
            != {name for name, _action, _view in animation_reviews.REVIEW_SPECS}
        ):
            raise contracts.ContractError("animation review record contract/hash is invalid")
        _verify_relative_artifact(root, review["contact_sheet"], label="contact sheet")
        for name, video in review["videos"].items():
            _verify_relative_artifact(root, video["video"], label=f"{name} video")
            _verify_relative_artifact(root, video["render_log"], label=f"{name} render log")
            _verify_relative_artifact(root, video["encode_log"], label=f"{name} encode log")
        reviews[asset_id] = {"payload": review, "path": review_path}
    if len(reviews) != payload["review_count"]:
        raise contracts.ContractError("animation review coverage is incomplete")
    return path, payload, reviews


def load_decisions(
    path: Path, review_batch: Mapping[str, Any], reviews: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"animation decisions are missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "animation_review_batch_sha256", "reviewer", "decisions"}
        or payload.get("schema") != DECISIONS_SCHEMA
        or payload.get("animation_review_batch_sha256") != review_batch["batch_sha256"]
        or not isinstance(payload.get("reviewer"), str)
        or not payload["reviewer"].strip()
        or not isinstance(payload.get("decisions"), list)
    ):
        raise contracts.ContractError("animation decisions contract/batch hash is invalid")
    expected_fields = {
        "asset_id",
        "review_sha256",
        "decision",
        "checks",
        "caveats",
        "notes",
    }
    decisions: dict[str, Any] = {}
    for decision in payload["decisions"]:
        if not isinstance(decision, dict) or set(decision) != expected_fields:
            raise contracts.ContractError("animation decision fields are invalid")
        asset_id = decision["asset_id"]
        if asset_id in decisions or asset_id not in reviews:
            raise contracts.ContractError("animation decision identity is invalid")
        if decision["review_sha256"] != reviews[asset_id]["payload"]["review_sha256"]:
            raise contracts.ContractError("animation decision review hash changed")
        if decision["decision"] not in {"approved_for_ue_apartment", "rejected"}:
            raise contracts.ContractError("invalid animation decision")
        checks = decision["checks"]
        if (
            not isinstance(checks, dict)
            or set(checks) != CHECK_FIELDS
            or any(not isinstance(value, bool) for value in checks.values())
        ):
            raise contracts.ContractError("animation decision checks are invalid")
        if decision["decision"] == "approved_for_ue_apartment" and not all(checks.values()):
            raise contracts.ContractError("approved animation decision has a failed check")
        if decision["decision"] == "rejected" and all(checks.values()):
            raise contracts.ContractError("rejected animation decision needs a failed check")
        caveats = decision["caveats"]
        if (
            not isinstance(caveats, list)
            or len(caveats) != len(set(caveats))
            or any(not isinstance(item, str) or not item for item in caveats)
            or not isinstance(decision["notes"], str)
            or not decision["notes"].strip()
        ):
            raise contracts.ContractError("animation decision notes/caveats are invalid")
        decisions[asset_id] = copy.deepcopy(decision)
    if set(decisions) != set(reviews):
        raise contracts.ContractError("animation decision coverage is incomplete")
    return payload


def publish_decisions(
    review_batch_path: Path, decisions_path: Path, output_root: Path
) -> Path:
    review_batch_path, review_batch, reviews = load_review_batch(review_batch_path)
    decisions = load_decisions(decisions_path, review_batch, reviews)
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        records = []
        for decision in sorted(decisions["decisions"], key=lambda item: item["asset_id"]):
            review = reviews[decision["asset_id"]]
            record: dict[str, Any] = {
                "schema": DECISION_SCHEMA,
                **copy.deepcopy(decision),
                "review": _record(review["path"]),
                "state_classification": (
                    "research_candidate"
                    if decision["decision"] == "approved_for_ue_apartment"
                    else "rejected"
                ),
                "formal_dataset_registration_authorized": False,
                "next_gate": (
                    "ue_import_metric_trajectory_audio_and_apartment_media"
                    if decision["decision"] == "approved_for_ue_apartment"
                    else "stop"
                ),
            }
            record["decision_sha256"] = _hash_without(record, "decision_sha256")
            destination = staging / decision["asset_id"] / "animation_decision.json"
            contracts.write_json_no_replace(destination, record)
            records.append(
                {
                    "asset_id": decision["asset_id"],
                    "decision": decision["decision"],
                    "decision_sha256": record["decision_sha256"],
                    "record": {
                        "path": destination.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                }
            )
        approved = sum(item["decision"] == "approved_for_ue_apartment" for item in records)
        batch: dict[str, Any] = {
            "schema": DECISION_BATCH_SCHEMA,
            "status": "completed_with_rejections" if approved != len(records) else "completed",
            "animation_review_batch": {
                "path": str(review_batch_path),
                "sha256": _sha256_file(review_batch_path),
                "batch_sha256": review_batch["batch_sha256"],
            },
            "decision_input": _record(Path(decisions_path)),
            "reviewer": decisions["reviewer"],
            "decision_count": len(records),
            "approved_count": approved,
            "rejected_count": len(records) - approved,
            "decisions": records,
            "automatic_checks": {
                "all_review_hashes_reauthenticated": True,
                "all_walk_idle_media_reauthenticated": True,
                "all_assets_have_one_decision": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        }
        batch["decision_batch_sha256"] = _hash_without(batch, "decision_batch_sha256")
        contracts.write_json_no_replace(
            staging / "animation_decision_batch_manifest.json", batch
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("animation decision output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "animation_decision_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animation-review-batch", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = publish_decisions(
            args.animation_review_batch, args.decisions, args.output_root
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError) as error:
        print(f"CONTROLLED_ANIMAL_ANIMATION_DECISION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_ANIMATION_DECISION_OK "
        f"approved={payload['approved_count']} rejected={payload['rejected_count']} "
        f"output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
