#!/usr/bin/env python3
"""Measure controlled quadruped dimensions from rigged mesh and UE evidence.

The normal entry point is standard-library only.  It launches Blender workers
to measure the weighted foreleg/shoulder surface in rig space, then calibrates
those ratios with the bounds that UE actually observed at the scene actor
scale.  Prompt targets are retained only for comparison and are never copied
into the measured result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
MEASUREMENT_SCHEMA = "controlled_animal_physical_measurement_v1"
BATCH_SCHEMA = "controlled_animal_physical_measurement_batch_v1"
METHOD = "ue_bounds_calibrated_weighted_foreleg_surface_v1"
FRONT_UPPER_GROUPS = frozenset({"Bone.014", "Bone.017"})


class MeasurementError(RuntimeError):
    """Raised when physical evidence is missing or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(
    path: Path, *, published_path: Path | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MeasurementError(f"artifact is not a direct non-empty file: {path}")
    return {
        "path": str(Path(published_path).resolve() if published_path else path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def quantile(values: Iterable[float], fraction: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers or not 0.0 <= fraction <= 1.0:
        raise MeasurementError("quantile requires values and a fraction in [0, 1]")
    if any(not math.isfinite(value) for value in numbers):
        raise MeasurementError("quantile values must be finite")
    position = (len(numbers) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def build_runtime_measurement(
    *,
    record: Mapping[str, Any],
    walking_spec: Mapping[str, Any],
    visual: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine independent mesh ratios with observed UE centimeter bounds."""

    tag = str(record["tag"])
    sources = walking_spec.get("sources")
    visual_sources = visual.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise MeasurementError(f"Walking spec must contain one source: {tag}")
    source = sources[0]
    matching = [item for item in visual_sources or [] if item.get("tag") == tag]
    if len(matching) != 1:
        raise MeasurementError(f"visual metadata must contain source {tag}")
    frames = matching[0].get("runtime_frames")
    if not isinstance(frames, list) or not frames:
        raise MeasurementError(f"visual metadata has no runtime frames: {tag}")
    if visual.get("automatic_checks", {}).get("overall") != "passed":
        raise MeasurementError(f"visual automatic checks did not pass: {tag}")

    actor_scale = float(source.get("actor_scale"))
    if not math.isfinite(actor_scale) or actor_scale <= 0.0:
        raise MeasurementError(f"invalid actor scale: {tag}")
    bounds_heights = []
    observed_scales = []
    for frame in frames:
        bounds = frame.get("bounds_ue", {})
        minimum = bounds.get("minimum_cm")
        maximum = bounds.get("maximum_cm")
        root_scale = frame.get("root_transform_ue", {}).get("scale")
        floor = frame.get("floor_contact", {})
        if (
            not isinstance(minimum, list)
            or not isinstance(maximum, list)
            or len(minimum) != 3
            or len(maximum) != 3
            or not isinstance(root_scale, list)
            or len(root_scale) != 3
            or floor.get("within_penetration_tolerance") is not True
        ):
            raise MeasurementError(f"incomplete UE frame evidence: {tag}")
        height = float(maximum[2]) - float(minimum[2])
        if not math.isfinite(height) or height <= 0.0:
            raise MeasurementError(f"invalid UE bounds height: {tag}")
        bounds_heights.append(height)
        observed_scales.extend(float(value) for value in root_scale)
    if max(abs(value - actor_scale) for value in observed_scales) > 1.0e-5:
        raise MeasurementError(f"UE actor scale changed across frames: {tag}")

    bounds_units = float(geometry.get("bounds_height_units"))
    shoulder_units = float(geometry.get("shoulder_height_units"))
    length_units = float(geometry.get("nose_to_tail_length_units"))
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (bounds_units, shoulder_units, length_units)
    ) or shoulder_units >= bounds_units:
        raise MeasurementError(f"invalid Blender geometry measurement: {tag}")

    bounds_height_cm = statistics.median(bounds_heights)
    centimeters_per_unit = bounds_height_cm / bounds_units
    shoulder_height_cm = shoulder_units * centimeters_per_unit
    nose_to_tail_length_cm = length_units * centimeters_per_unit
    target = record.get("target_physical_profile", {})
    if target.get("measurement") != "shoulder_height_cm":
        raise MeasurementError(f"unsupported physical measurement target: {tag}")
    target_cm = float(target.get("target_value_cm"))
    tolerance_cm = float(target.get("tolerance_cm"))
    residual_cm = shoulder_height_cm - target_cm
    comparison = (
        "within_tolerance" if abs(residual_cm) <= tolerance_cm else "outside_tolerance"
    )
    return {
        "schema": MEASUREMENT_SCHEMA,
        "asset_id": record["base_avatar_id"],
        "tag": tag,
        "profile_schema_id": record["profile_schema_id"],
        "sampled_size": record["sampled_attributes"][target["control_attribute"]],
        "method": METHOD,
        "physical_measurements": {
            "status": "measured",
            "method": METHOD,
            "runtime": {
                "actor_scale": round(actor_scale, 6),
                "shoulder_height_cm": round(shoulder_height_cm, 6),
                "bounds_height_cm": round(bounds_height_cm, 6),
                "nose_to_tail_length_cm": round(nose_to_tail_length_cm, 6),
            },
        },
        "target_comparison": {
            "target_value_cm": target_cm,
            "tolerance_cm": tolerance_cm,
            "measured_minus_target_cm": round(residual_cm, 6),
            "status": comparison,
            "target_reference_status": target.get("reference_provenance", {}).get(
                "status"
            ),
        },
        "calibration": {
            "ue_frame_count": len(frames),
            "ue_bounds_height_min_cm": round(min(bounds_heights), 6),
            "ue_bounds_height_median_cm": round(bounds_height_cm, 6),
            "ue_bounds_height_max_cm": round(max(bounds_heights), 6),
            "centimeters_per_blender_unit": round(centimeters_per_unit, 9),
            "geometry": dict(geometry),
        },
    }


def summarize_size_ordering(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        grouped.setdefault(str(record["profile_schema_id"]), {}).setdefault(
            str(record["sampled_size"]), []
        ).append(
            float(record["physical_measurements"]["runtime"]["shoulder_height_cm"])
        )
    summaries = []
    for profile_id, values_by_size in sorted(grouped.items()):
        ordered = [size for size in ("small", "medium", "large") if size in values_by_size]
        medians = {
            size: round(statistics.median(values_by_size[size]), 6) for size in ordered
        }
        strictly_increasing = all(
            medians[left] < medians[right]
            for left, right in zip(ordered, ordered[1:])
        )
        summaries.append(
            {
                "profile_schema_id": profile_id,
                "ordered_sizes": ordered,
                "median_shoulder_height_cm": medians,
                "strictly_increasing": strictly_increasing,
                "status": "passed" if strictly_increasing else "rejected",
            }
        )
    return summaries


def _blender_geometry_measurement(input_glb: Path, output: Path) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    if output.exists() or output.is_symlink():
        raise MeasurementError(f"worker output already exists: {output}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise MeasurementError("rigged GLB contains no mesh")
    mesh = max(meshes, key=lambda obj: len(obj.data.vertices))
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    missing = FRONT_UPPER_GROUPS - set(group_names.values())
    if missing:
        raise MeasurementError(f"missing foreleg vertex groups: {sorted(missing)}")

    all_x: list[float] = []
    all_z: list[float] = []
    shoulder_z: list[float] = []
    for vertex in mesh.data.vertices:
        point = mesh.matrix_world @ Vector(vertex.co)
        all_x.append(float(point.x))
        all_z.append(float(point.z))
        weight = sum(
            float(membership.weight)
            for membership in vertex.groups
            if group_names.get(membership.group) in FRONT_UPPER_GROUPS
        )
        if weight >= 0.05:
            shoulder_z.append(float(point.z))
    if len(shoulder_z) < 100:
        raise MeasurementError(
            f"too few weighted shoulder vertices: {len(shoulder_z)}"
        )
    floor_z = quantile(all_z, 0.001)
    top_z = quantile(all_z, 0.999)
    shoulder_surface_z = quantile(shoulder_z, 0.95)
    min_x = quantile(all_x, 0.005)
    max_x = quantile(all_x, 0.995)
    geometry = {
        "schema": "weighted_quadruped_geometry_measurement_v1",
        "input_glb": _artifact(input_glb),
        "mesh_name": mesh.name,
        "vertex_count": len(mesh.data.vertices),
        "front_upper_groups": sorted(FRONT_UPPER_GROUPS),
        "selected_shoulder_vertex_count": len(shoulder_z),
        "quantiles": {
            "floor": 0.001,
            "top": 0.999,
            "shoulder_surface": 0.95,
            "length_min": 0.005,
            "length_max": 0.995,
        },
        "bounds_height_units": top_z - floor_z,
        "shoulder_height_units": shoulder_surface_z - floor_z,
        "nose_to_tail_length_units": max_x - min_x,
        "shoulder_fraction_of_bounds_height": (shoulder_surface_z - floor_z)
        / (top_z - floor_z),
    }
    if not (
        0.2 < geometry["shoulder_fraction_of_bounds_height"] < 0.98
        and geometry["nose_to_tail_length_units"] > 0.0
    ):
        raise MeasurementError("implausible weighted shoulder geometry")
    _atomic_json(output, geometry)


def _load_records(manifest_paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in manifest_paths:
        payload = _read_json(path)
        if (
            payload.get("schema") != SCHEMA
            or payload.get("avatar_count") != len(payload.get("records", []))
            or payload.get("clip_count") != len(payload.get("records", [])) * 2
        ):
            raise MeasurementError(f"invalid controlled animal manifest: {path}")
        for record in payload["records"]:
            asset_id = record.get("base_avatar_id")
            if not asset_id or asset_id in seen:
                raise MeasurementError(f"duplicate/empty controlled asset: {asset_id}")
            seen.add(asset_id)
            records.append(record)
    return sorted(records, key=lambda item: item["base_avatar_id"])


def build_measurements(
    *,
    manifest_paths: Sequence[Path],
    output_root: Path,
    blender: Path,
    workers: int,
) -> Path:
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise MeasurementError(f"output root already exists: {output_root}")
    if not 1 <= workers <= 8:
        raise MeasurementError("workers must be in [1, 8]")
    records = _load_records(manifest_paths)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent)
    )
    try:
        geometry_paths = {
            record["base_avatar_id"]: staging
            / "geometry"
            / f"{record['base_avatar_id']}.json"
            for record in records
        }
        for record in records:
            source = Path(record["source_glb"]["path"]).resolve()
            if (
                not source.is_file()
                or source.is_symlink()
                or _sha256(source) != record["source_glb"]["sha256"]
            ):
                raise MeasurementError(
                    f"source GLB changed: {record['base_avatar_id']}"
                )

        def run_worker(record: Mapping[str, Any]) -> tuple[str, subprocess.CompletedProcess[str]]:
            asset_id = str(record["base_avatar_id"])
            output = geometry_paths[asset_id]
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(blender),
                "-b",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                "--blender-worker",
                "--input-glb",
                str(Path(record["source_glb"]["path"]).resolve()),
                "--output",
                str(output),
            ]
            return asset_id, subprocess.run(
                command,
                cwd=SPEAR_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_worker, record): record for record in records}
            for future in as_completed(futures):
                asset_id, process = future.result()
                log = staging / "logs" / f"{asset_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0 or not geometry_paths[asset_id].is_file():
                    failures.append(
                        {"asset_id": asset_id, "returncode": process.returncode, "log": str(log)}
                    )
        if failures:
            raise MeasurementError(f"Blender physical measurement failed: {failures}")

        measured = []
        for record in records:
            asset_id = record["base_avatar_id"]
            action = record["actions"]["Walking"]
            walking_spec = _read_json(Path(action["spec"]))
            visual_path = (
                Path(action["output_dir"]) / "videos" / "actor_visual_metadata.json"
            )
            visual = _read_json(visual_path)
            geometry = _read_json(geometry_paths[asset_id])
            result = build_runtime_measurement(
                record=record,
                walking_spec=walking_spec,
                visual=visual,
                geometry=geometry,
            )
            result["evidence"] = {
                "walking_spec": _artifact(Path(action["spec"])),
                "ue_visual_metadata": _artifact(visual_path),
                "geometry_measurement": _artifact(
                    geometry_paths[asset_id],
                    published_path=output_root / "geometry" / f"{asset_id}.json",
                ),
            }
            destination = staging / "measurements" / f"{asset_id}.json"
            _atomic_json(destination, result)
            measured.append(result)
        ordering = summarize_size_ordering(measured)
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "asset_count": len(measured),
            "inputs": [_artifact(Path(path)) for path in manifest_paths],
            "measurements": [
                {
                    "asset_id": item["asset_id"],
                    "profile_schema_id": item["profile_schema_id"],
                    "sampled_size": item["sampled_size"],
                    "physical_measurements": item["physical_measurements"],
                    "target_comparison": item["target_comparison"],
                    "record": _artifact(
                        staging / "measurements" / f"{item['asset_id']}.json",
                        published_path=output_root
                        / "measurements"
                        / f"{item['asset_id']}.json",
                    ),
                }
                for item in measured
            ],
            "size_ordering": ordering,
            "automatic_checks": {
                "all_measurements_derived_from_mesh_and_ue_evidence": True,
                "all_profile_size_medians_strictly_increasing": all(
                    item["strictly_increasing"] for item in ordering
                ),
                "within_provisional_target_tolerance_count": sum(
                    item["target_comparison"]["status"] == "within_tolerance"
                    for item in measured
                ),
                "overall": "passed"
                if all(item["strictly_increasing"] for item in ordering)
                else "rejected",
            },
        }
        manifest["batch_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _atomic_json(staging / "measurement_batch_manifest.json", manifest)
        os.rename(staging, output_root)
        return output_root / "measurement_batch_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--blender", type=Path, default=Path("/data/jzy/.local/bin/blender"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--input-glb", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    args = _parse_args(argv)
    try:
        if args.blender_worker:
            if args.input_glb is None or args.output is None:
                raise MeasurementError("Blender worker requires --input-glb and --output")
            _blender_geometry_measurement(args.input_glb.resolve(), args.output.resolve())
            print(f"CONTROLLED_ANIMAL_GEOMETRY_MEASUREMENT_OK output={args.output.resolve()}")
            return 0
        if not args.manifest or args.output_root is None:
            raise MeasurementError("batch mode requires --manifest and --output-root")
        manifest = build_measurements(
            manifest_paths=args.manifest,
            output_root=args.output_root,
            blender=args.blender,
            workers=args.workers,
        )
        payload = _read_json(manifest)
    except (MeasurementError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_FAILED {error}", flush=True)
        return 2
    print(
        f"CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_OK assets={payload['asset_count']} "
        f"overall={payload['automatic_checks']['overall']} output={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
