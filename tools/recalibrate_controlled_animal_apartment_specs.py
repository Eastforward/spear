#!/usr/bin/env python3
"""Publish immutable Apartment specs corrected by measured animal dimensions.

The input target is only a provisional semantic control.  The scale correction
therefore uses the actor scale and shoulder height observed by UE, then creates
a new review batch instead of overwriting already-rendered evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
MEASUREMENT_SCHEMA = "controlled_animal_physical_measurement_batch_v1"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def calibrated_actor_scale(measurement: dict) -> float:
    runtime = measurement["physical_measurements"]["runtime"]
    old_scale = float(runtime["actor_scale"])
    observed = float(runtime["shoulder_height_cm"])
    target = float(measurement["target_comparison"]["target_value_cm"])
    if old_scale <= 0.0 or observed <= 0.0 or target <= 0.0:
        raise ValueError("scale calibration inputs must be positive")
    return round(old_scale * target / observed, 6)


def relative_target_error(measurement: dict) -> float:
    observed = float(
        measurement["physical_measurements"]["runtime"]["shoulder_height_cm"]
    )
    target = float(measurement["target_comparison"]["target_value_cm"])
    if target <= 0.0:
        raise ValueError("target shoulder height must be positive")
    return abs(observed - target) / target


def _calibrate_spec(spec: dict, *, measurement: dict, measurement_batch: dict) -> dict:
    result = copy.deepcopy(spec)
    scale = calibrated_actor_scale(measurement)
    runtime = measurement["physical_measurements"]["runtime"]
    comparison = measurement["target_comparison"]
    source = result["sources"][0]
    source["actor_scale"] = scale

    rationale = result["camera_pass_table_loop_contract"]["animal_scale_rationale"]
    base_scale = float(rationale.get("base_actor_scale", runtime["actor_scale"]))
    rationale.update(
        {
            "actor_scale": scale,
            "physical_scale_ratio": round(scale / base_scale, 6),
            "policy": (
                "post-measurement per-instance UE scale calibration to the "
                "provisional shoulder-height target"
            ),
            "measurement_feedback": {
                "measurement_batch": measurement_batch,
                "observed_actor_scale": float(runtime["actor_scale"]),
                "observed_shoulder_height_cm": float(runtime["shoulder_height_cm"]),
                "relative_target_error": round(relative_target_error(measurement), 6),
                "target_shoulder_height_cm": float(comparison["target_value_cm"]),
            },
        }
    )
    assertions = result.setdefault("runtime_assertions", {})
    assertions["ground_snap_max_abs_correction_cm"] = round(
        max(25.0, scale * 200.0), 6
    )
    assertions["ground_snap_to_floor"] = True
    return result


def build_recalibration_batch(
    *,
    manifest_path: Path,
    measurement_batch_path: Path,
    output_root: Path,
    max_relative_error: float,
    asset_ids: set[str] | None = None,
) -> Path:
    manifest_path = manifest_path.resolve()
    measurement_batch_path = measurement_batch_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if max_relative_error < 0.0:
        raise ValueError("max_relative_error must be non-negative")

    source = _read_json(manifest_path)
    measurements = _read_json(measurement_batch_path)
    if source.get("schema") != SCHEMA:
        raise ValueError(f"unexpected Apartment manifest schema: {source.get('schema')}")
    if measurements.get("schema") != MEASUREMENT_SCHEMA:
        raise ValueError(
            f"unexpected measurement schema: {measurements.get('schema')}"
        )
    by_asset = {item["asset_id"]: item for item in measurements["measurements"]}
    if len(by_asset) != len(measurements["measurements"]):
        raise ValueError("duplicate asset_id in measurement batch")

    selected = []
    requested = set(asset_ids or ())
    for record in source["records"]:
        asset_id = record["asset_id"]
        measurement = by_asset.get(asset_id)
        if measurement is None:
            continue
        if requested and asset_id not in requested:
            continue
        if measurement["target_comparison"]["status"] != "outside_tolerance":
            continue
        if relative_target_error(measurement) <= max_relative_error:
            continue
        selected.append((record, measurement))
    if requested - {item[0]["asset_id"] for item in selected}:
        raise ValueError(
            "requested assets were not eligible for recalibration: "
            f"{sorted(requested - {item[0]['asset_id'] for item in selected})}"
        )
    if not selected:
        raise ValueError("no assets exceeded the requested relative target error")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        measurement_artifact = _artifact(measurement_batch_path)
        records = []
        for original_record, measurement in selected:
            record = copy.deepcopy(original_record)
            record["actions"] = {}
            asset_id = record["asset_id"]
            for action, original_action in sorted(original_record["actions"].items()):
                original_spec_path = Path(original_action["spec"]).resolve()
                spec = _calibrate_spec(
                    _read_json(original_spec_path),
                    measurement=measurement,
                    measurement_batch=measurement_artifact,
                )
                suffix = original_spec_path.name
                staged_spec = staging / "specs" / f"pixal_{asset_id}" / suffix
                _write_json(staged_spec, spec)
                published_spec = output_root / staged_spec.relative_to(staging)

                original_output = Path(original_action["output_dir"])
                output_suffix = original_output.name
                published_output = output_root / "clips" / f"pixal_{asset_id}" / output_suffix
                record["actions"][action] = {
                    "clip_id": f"{original_action['clip_id']}_scale_calibrated_v1",
                    "motion": original_action["motion"],
                    "output_dir": str(published_output),
                    "spec": str(published_spec),
                    "spec_evidence": _artifact(staged_spec)
                    | {"path": str(published_spec)},
                }
            record["scale_recalibration"] = {
                "new_actor_scale": calibrated_actor_scale(measurement),
                "observed_shoulder_height_cm": measurement["physical_measurements"][
                    "runtime"
                ]["shoulder_height_cm"],
                "relative_target_error": round(relative_target_error(measurement), 6),
                "target_shoulder_height_cm": measurement["target_comparison"][
                    "target_value_cm"
                ],
            }
            records.append(record)

        published = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "avatar_count": len(records),
            "clip_count": sum(len(item["actions"]) for item in records),
            "audio_policy": source.get("audio_policy"),
            "trajectory_policy": source.get("trajectory_policy"),
            "physical_scale_policy": (
                "per-instance actor scale corrected from UE-observed shoulder height"
            ),
            "inputs": {
                "source_apartment_manifest": _artifact(manifest_path),
                "measurement_batch": measurement_artifact,
            },
            "selection": {
                "max_relative_error": max_relative_error,
                "requested_asset_ids": sorted(requested),
                "selected_asset_ids": [item["asset_id"] for item in records],
            },
            "records": records,
        }
        published["manifest_sha256"] = _canonical_sha256(published)
        staged_manifest = staging / "spec_manifest.json"
        _write_json(staged_manifest, published)
        os.rename(staging, output_root)
        return output_root / "spec_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measurement-batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-relative-error", type=float, default=0.5)
    parser.add_argument("--asset-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = build_recalibration_batch(
            manifest_path=args.manifest,
            measurement_batch_path=args.measurement_batch,
            output_root=args.output_root,
            max_relative_error=args.max_relative_error,
            asset_ids=set(args.asset_id),
        )
    except Exception as error:
        print(
            "CONTROLLED_ANIMAL_APARTMENT_SCALE_RECALIBRATION_FAILED "
            f"{type(error).__name__}: {error}"
        )
        return 2
    published = _read_json(result)
    print(
        "CONTROLLED_ANIMAL_APARTMENT_SCALE_RECALIBRATION_OK "
        f"assets={published['avatar_count']} clips={published['clip_count']} "
        f"output={result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
